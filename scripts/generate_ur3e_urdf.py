"""
UR3e URDF 生成スクリプト（Dockerfile から呼ばれる）

1. ur_description/urdf/ の xacro ファイル中の $(find ur_description) を絶対パスに置換
2. xacro で UR3e アーム URDF を生成
3. シンプルな並列グリッパー定義を末尾に追加
4. /opt/ur3e/ur3e_with_gripper.urdf に保存
"""

import subprocess
import sys
from pathlib import Path

UR_DESC_ROOT = Path("/opt/universal_robot/ur_description")
OUT_DIR = Path("/opt/ur3e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. $(find ur_description) → 絶対パスに置換 ───────────────────────────────
#  melodic-devel ブランチの xacro は ROS1 マクロを使うため、
#  スタンドアロン実行用にパスを事前にパッチする
print("[info] Patching $(find ur_description) in xacro files ...")
for xacro_file in UR_DESC_ROOT.rglob("*.xacro"):
    content = xacro_file.read_text(errors="replace")
    patched = content.replace("$(find ur_description)", str(UR_DESC_ROOT))
    if patched != content:
        xacro_file.write_text(patched)
        print(f"  patched: {xacro_file.relative_to(UR_DESC_ROOT)}")

# ── 2. xacro で UR3e アームを生成 ────────────────────────────────────────────
xacro_candidates = [
    UR_DESC_ROOT / "urdf" / "ur3e.xacro",
    UR_DESC_ROOT / "urdf" / "ur3e.urdf.xacro",
    UR_DESC_ROOT / "urdf" / "ur3e_robot.urdf.xacro",
]

xacro_file = None
for c in xacro_candidates:
    if c.exists():
        xacro_file = c
        break

if xacro_file is None:
    print("[ERROR] UR3e xacro not found. Searched:")
    for c in xacro_candidates:
        print(f"  {c}")
    print("Available files in urdf/:")
    for f in (UR_DESC_ROOT / "urdf").iterdir():
        print(f"  {f.name}")
    sys.exit(1)

print(f"[info] Processing xacro: {xacro_file}")

result = subprocess.run(
    ["xacro", str(xacro_file)],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(f"[ERROR] xacro failed:\n{result.stderr}")
    sys.exit(1)

arm_urdf = result.stdout

# ── 2b. package://ur_description → 絶対パスに置換 ────────────────────────────
#  xacro 出力に残る package://ur_description を絶対パスに変換し
#  ROS なし環境でも SAPIEN がメッシュを解決できるようにする
arm_urdf = arm_urdf.replace("package://ur_description", str(UR_DESC_ROOT))
print(f"[info] Replaced package://ur_description with {UR_DESC_ROOT}")

# ── 3. シンプルな並列グリッパー追加 ──────────────────────────────────────────
GRIPPER_URDF = """
  <!-- ========================================================== -->
  <!-- Simple parallel gripper attached at tool0                  -->
  <!-- ========================================================== -->

  <link name="gripper_base_link">
    <visual>
      <origin xyz="0 0 0.015" rpy="0 0 0"/>
      <geometry><box size="0.06 0.06 0.03"/></geometry>
      <material name="dark_grey"><color rgba="0.3 0.3 0.3 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.015" rpy="0 0 0"/>
      <geometry><box size="0.06 0.06 0.03"/></geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>

  <joint name="gripper_base_joint" type="fixed">
    <parent link="tool0"/>
    <child link="gripper_base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Left finger -->
  <link name="left_finger_link">
    <visual>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><box size="0.01 0.02 0.08"/></geometry>
      <material name="dark_grey"><color rgba="0.3 0.3 0.3 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><box size="0.01 0.02 0.08"/></geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <joint name="left_finger_joint" type="prismatic">
    <parent link="gripper_base_link"/>
    <child link="left_finger_link"/>
    <origin xyz="-0.02 0 0.03" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.04" effort="20" velocity="0.1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Right finger (mimic) -->
  <link name="right_finger_link">
    <visual>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><box size="0.01 0.02 0.08"/></geometry>
      <material name="dark_grey"><color rgba="0.3 0.3 0.3 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><box size="0.01 0.02 0.08"/></geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <joint name="right_finger_joint" type="prismatic">
    <parent link="gripper_base_link"/>
    <child link="right_finger_link"/>
    <origin xyz="0.02 0 0.03" rpy="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit lower="0.0" upper="0.04" effort="20" velocity="0.1"/>
    <dynamics damping="0.1" friction="0.0"/>
    <mimic joint="left_finger_joint" multiplier="1.0" offset="0.0"/>
  </joint>

  <!-- TCP frame (between fingertips) -->
  <link name="gripper_tcp_link"/>
  <joint name="gripper_tcp_joint" type="fixed">
    <parent link="gripper_base_link"/>
    <child link="gripper_tcp_link"/>
    <origin xyz="0 0 0.11" rpy="0 0 0"/>
  </joint>
"""

combined_urdf = arm_urdf.replace("</robot>", GRIPPER_URDF + "\n</robot>")

out_path = OUT_DIR / "ur3e_with_gripper.urdf"
out_path.write_text(combined_urdf)
print(f"[OK] UR3e URDF written: {out_path}")

# ── 4. 簡単な検証 ─────────────────────────────────────────────────────────────
try:
    import yourdfpy
    robot = yourdfpy.URDF.load(str(out_path))
    joints = list(robot.joint_map.keys())
    print(f"[OK] URDF validated. Joints ({len(joints)}): {joints}")
except ImportError:
    print("[skip] yourdfpy not installed, skipping validation")
except Exception as e:
    print(f"[WARN] URDF validation: {e}")
