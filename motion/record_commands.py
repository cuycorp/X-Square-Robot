#!/usr/bin/env python3
"""Record the robot's OWN arm commands while someone operates it, to learn the command format. READ-ONLY.

It only subscribes (never publishes) to:
  /{side}_arm_cartesian_controller/pose_cmd   what the platform sends to the arm controller
  /{side}_arm/end_pose                         where the arm actually is
  /{side}_gripper_controller/commands          gripper commands
  /{side}_gripper/joint_states                 gripper position
and then answers: which frame does pose_cmd use, at what rate, how big is each step, and what do gripper
commands look like.

Usage (start it, then have someone move the arm with the leader arm / teleop, or start a Wall-X run):
  .venv/bin/python motion/record_commands.py --seconds 15
  .venv/bin/python motion/record_commands.py --analyze .cache/recordings/<file>.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kinematics as K  # noqa: E402

ROSBRIDGE_URL = "ws://192.168.77.2:9090"
OUT_DIR = HERE.parent / ".cache" / "recordings"
URDF_CACHE = HERE.parent / ".cache" / "robot_description.urdf"

TOPICS = {}
for _s in K.ARMS:
    TOPICS[f"/{_s}_arm_cartesian_controller/pose_cmd"] = "geometry_msgs/msg/PoseStamped"
    TOPICS[f"/{_s}_arm/end_pose"] = "geometry_msgs/msg/PoseStamped"
    TOPICS[f"/{_s}_arm/joint_states"] = "sensor_msgs/msg/JointState"   # faster than end_pose; same pose via FK
    TOPICS[f"/{_s}_gripper_controller/commands"] = "std_msgs/msg/Float64MultiArray"
    TOPICS[f"/{_s}_gripper/joint_states"] = "sensor_msgs/msg/JointState"


def record(seconds: float, url: str = ROSBRIDGE_URL) -> Path:
    from websockets.sync.client import connect
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / time.strftime("commands-%Y%m%d-%H%M%S.jsonl")
    counts = {t: 0 for t in TOPICS}
    with connect(url, open_timeout=5, max_size=None) as ws, path.open("w") as f:
        for t, typ in TOPICS.items():
            ws.send(json.dumps({"op": "subscribe", "id": f"rec{t}", "topic": t, "type": typ, "queue_length": 50}))
        start = time.monotonic()
        last_print = start
        while (now := time.monotonic()) - start < seconds:
            try:
                m = json.loads(ws.recv(timeout=0.5))
            except TimeoutError:
                continue
            if m.get("op") == "publish" and m.get("topic") in TOPICS:
                f.write(json.dumps({"t": now - start, "topic": m["topic"], "msg": m["msg"]}) + "\n")
                counts[m["topic"]] += 1
            if now - last_print > 2:
                cmds = sum(v for k, v in counts.items() if "commands" in k or "pose_cmd" in k)
                print(f"  {now - start:4.0f} s: {cmds} command messages so far", flush=True)
                last_print = now
        for t in TOPICS:
            ws.send(json.dumps({"op": "unsubscribe", "id": f"rec{t}", "topic": t}))
    return path


def _to_dict(msg, msgtype: str) -> dict:
    """The fields we analyse, from a deserialized rosbags message."""
    if msgtype.endswith("PoseStamped"):
        p, o = msg.pose.position, msg.pose.orientation
        return {"header": {"frame_id": msg.header.frame_id},
                "pose": {"position": {"x": p.x, "y": p.y, "z": p.z}, "orientation": {"x": o.x, "y": o.y, "z": o.z, "w": o.w}}}
    if msgtype.endswith("JointState"):
        return {"name": list(msg.name), "position": [float(v) for v in msg.position]}
    if msgtype.endswith("Float64MultiArray"):
        return {"data": [float(v) for v in msg.data]}
    raise ValueError(msgtype)


def bag_topics(path: Path) -> list:
    """[(topic, msgtype, count)] of a rosbag2 directory or .mcap file. Read-only, local file."""
    from rosbags.highlevel import AnyReader
    with AnyReader([path]) as reader:
        return sorted((c.topic, c.msgtype, c.msgcount) for c in reader.connections)


def load_bag(path: Path) -> list:
    """Rows in the recorder's format from a robot recording (only the topics we analyse)."""
    from rosbags.highlevel import AnyReader
    rows, t0 = [], None
    with AnyReader([path]) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, stamp, raw in reader.messages(connections=conns):
            t0 = stamp if t0 is None else t0
            try:
                msg = _to_dict(reader.deserialize(raw, conn.msgtype), conn.msgtype)
            except ValueError:
                continue
            rows.append({"t": (stamp - t0) / 1e9, "topic": conn.topic, "msg": msg})
    return rows


def _pose(msg):
    p, o = msg["pose"]["position"], msg["pose"]["orientation"]
    return np.array([p["x"], p["y"], p["z"]]), K.quat_to_matrix(o["x"], o["y"], o["z"], o["w"]), msg["header"]["frame_id"]


def compare_streams(cmd, actual, lags=np.arange(-0.10, 0.51, 0.01)):
    """cmd/actual: lists of (t, position). The actual position is interpolated at each command time + lag
    (rosbridge relays both streams at a few Hz, so exact time matches are rare). Returns the best lag and
    position error stats."""
    if len(cmd) < 3 or len(actual) < 2:
        return None
    actual = sorted(actual, key=lambda a: a[0])
    ta = np.array([a[0] for a in actual])
    pa = np.array([a[1] for a in actual])
    tc = np.array([c[0] for c in cmd])
    pc = np.array([c[1] for c in cmd])
    best = None
    for lag in lags:
        t = tc + lag
        ok = (t >= ta[0]) & (t <= ta[-1])
        if ok.sum() < max(3, 0.3 * len(tc)):
            continue
        interp = np.stack([np.interp(t[ok], ta, pa[:, k]) for k in range(3)], axis=1)
        err = np.linalg.norm(pc[ok] - interp, axis=1)
        med = float(np.median(err))
        if best is None or med < best["median_mm"] / 1000:
            best = {"lag_ms": round(1000 * float(lag)), "median_mm": 1000 * med, "p95_mm": 1000 * float(np.percentile(err, 95)),
                    "n": int(ok.sum())}
    return best


def analyze(path: Path) -> dict:
    if path.is_dir() and not (path / "metadata.yaml").exists():
        return {"file": str(path), "duration_s": 0, "topics": {}, "arms": {s: {} for s in K.ARMS},
                "verdict": ["incomplete recording (no metadata.yaml); skipped"]}
    if path.is_dir() or path.suffix == ".mcap":
        rows = load_bag(path)
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"file": str(path), "duration_s": 0, "topics": {}, "arms": {s: {} for s in K.ARMS},
                "verdict": ["the recording contains none of the arm command/pose topics"]}
    report = {"file": str(path), "duration_s": round(rows[-1]["t"], 1) if rows else 0, "topics": {}, "arms": {}}
    by_topic = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)
    for t, rs in by_topic.items():
        span = rs[-1]["t"] - rs[0]["t"]
        report["topics"][t] = {"messages": len(rs), "rate_hz": round((len(rs) - 1) / span, 1) if span > 0 else None}
    urdf = URDF_CACHE.read_text() if URDF_CACHE.exists() else None
    for s in K.ARMS:
        arm = {}
        cmds = [(r["t"], *_pose(r["msg"])) for r in by_topic.get(f"/{s}_arm_cartesian_controller/pose_cmd", [])]
        acts = [(r["t"], *_pose(r["msg"])) for r in by_topic.get(f"/{s}_arm/end_pose", [])]
        if urdf:  # add joint_states -> FK in the end_pose convention (validated to 0.0 mm)
            chain = K.Arm(urdf, s)
            N_inv = np.linalg.inv(chain.fk(np.zeros(6)))
            for r in by_topic.get(f"/{s}_arm/joint_states", []):
                pos = dict(zip(r["msg"]["name"], r["msg"]["position"]))
                if all(n in pos for n in chain.names):
                    T = N_inv @ chain.fk([pos[n] for n in chain.names])
                    acts.append((r["t"], T[:3, 3], T[:3, :3], f"{s}_arm_neutral (from joints)"))
        arm["actual_samples"] = len(acts)
        if cmds:
            arm["pose_cmd_frames"] = sorted({c[3] for c in cmds})
            steps = [1000 * np.linalg.norm(cmds[i][1] - cmds[i - 1][1]) for i in range(1, len(cmds))]
            arm["step_mm"] = {"median": round(float(np.median(steps)), 2), "max": round(float(np.max(steps)), 2)} if steps else None
            arm["pose_cmd_range_m"] = {"min": np.round(np.min([c[1] for c in cmds], axis=0), 4).tolist(),
                                       "max": np.round(np.max([c[1] for c in cmds], axis=0), 4).tolist()}
            # Hypothesis A: pose_cmd is in the same frame as end_pose (relative to the neutral pose).
            arm["match_as_end_pose_frame"] = compare_streams([(c[0], c[1]) for c in cmds], [(a[0], a[1]) for a in acts])
            # Hypothesis B: pose_cmd is the absolute link6 pose in the arm base frame.
            if urdf and acts:
                N = K.Arm(urdf, s).fk(np.zeros(6))
                abs_acts = [(a[0], (N @ K.transform(a[2], a[1]))[:3, 3]) for a in acts]
                arm["match_as_arm_base_frame"] = compare_streams([(c[0], c[1]) for c in cmds], abs_acts)
        g_cmd = [r["msg"]["data"] for r in by_topic.get(f"/{s}_gripper_controller/commands", [])]
        g_pos = [r["msg"]["position"][0] for r in by_topic.get(f"/{s}_gripper/joint_states", []) if r["msg"].get("position")]
        if g_cmd:
            flat = [v for d in g_cmd for v in d]
            arm["gripper_cmd"] = {"messages": len(g_cmd), "values_per_msg": len(g_cmd[0]),
                                  "min": round(min(flat), 3), "max": round(max(flat), 3)}
        if g_pos:
            arm["gripper_position"] = {"min": round(min(g_pos), 3), "max": round(max(g_pos), 3),
                                       "first": round(g_pos[0], 3), "last": round(g_pos[-1], 3)}
        report["arms"][s] = arm
    report["verdict"] = verdict(report)
    return report


MIN_MOTION_M = 0.02   # commands must span at least this much for a match to prove anything


def verdict(report) -> list:
    lines = []
    for s, a in report["arms"].items():
        if "pose_cmd_frames" not in a:
            lines.append(f"{s}: no pose_cmd messages — nobody drove this arm through the cartesian controller during the recording.")
            continue
        span = float(np.max(np.subtract(a["pose_cmd_range_m"]["max"], a["pose_cmd_range_m"]["min"])))
        if span < MIN_MOTION_M:
            A, B = a.get("match_as_end_pose_frame"), a.get("match_as_arm_base_frame")
            hint = (f" (static hint: a constant command matched the resting arm at {A['median_mm']:.0f} mm as end_pose-style vs "
                    f"{B['median_mm']:.0f} mm as arm-base-style)") if A and B else ""
            lines.append(f"{s}: INCONCLUSIVE: the commands barely moved ({100 * span:.1f} cm span), so a match proves nothing"
                         f"{hint}. Need a recording where the arm moves.")
            continue
        A, B = a.get("match_as_end_pose_frame"), a.get("match_as_arm_base_frame")
        if A and (not B or A["median_mm"] < B["median_mm"]) and A["median_mm"] < 10:
            lines.append(f"{s}: pose_cmd uses the end_pose convention (frame {a['pose_cmd_frames']}): the arm follows it within "
                         f"{A['median_mm']:.1f} mm median ({A['p95_mm']:.1f} mm p95) about {A['lag_ms']} ms later. "
                         f"Steps are {a['step_mm']['median']} mm median per message. → our preview format is CONFIRMED.")
        elif B and B["median_mm"] < 10:
            lines.append(f"{s}: pose_cmd is the ABSOLUTE pose in the arm base frame ({B['median_mm']:.1f} mm median). "
                         f"→ our preview format is WRONG; the sender must add FK(0).")
        else:
            best = min([x for x in (A, B) if x], key=lambda x: x["median_mm"], default=None)
            lines.append(f"{s}: pose_cmd does not clearly match either convention (best median "
                         f"{best['median_mm']:.0f} mm). → do NOT send anything; investigate further." if best else
                         f"{s}: not enough end_pose data to compare.")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--analyze", metavar="FILE", help="analyse an existing recording (.jsonl, or a robot rosbag dir/.mcap)")
    ap.add_argument("--topics", metavar="BAG", help="list the topics in a robot rosbag (dir or .mcap)")
    a = ap.parse_args()
    if a.topics:
        for topic, typ, n in bag_topics(Path(a.topics)):
            mark = "  <-- used" if topic in TOPICS else ""
            print(f"{n:8d}  {topic}  [{typ}]{mark}")
        return 0
    if a.analyze:
        path = Path(a.analyze)
    else:
        print(f"Recording for {a.seconds:.0f} s (read-only). Move the arm with the leader arm / teleop now…", flush=True)
        path = record(a.seconds)
        print(f"Saved {path}")
    report = analyze(path)
    print(json.dumps({k: v for k, v in report.items() if k != "verdict"}, indent=2))
    print("\nVERDICT")
    for line in report["verdict"]:
        print(" -", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
