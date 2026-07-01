#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.paths import package_root, resource_path


HARDWARE_DIR = ROOT / "config" / "hardware"
HARDWARE_LAUNCHES = (
    "hri_chess_pre.launch",
    "data_collection.launch",
    "eva_chess.launch",
)
RUNTIME_SCRIPTS = (
    "chess_commander.py",
    "chess_robo_player.py",
    "chess_robot_recovery.py",
    "data_collect_chess_commander.py",
    "data_collect_chess_robot.py",
    "eva_chess_commander.py",
    "eva_chess_robot.py",
    "llm_chess_commander.py",
    "llm_commentary_manager.py",
    "hri_chess_commander.py",
    "hri_chess_robot.py",
    "ros_param_manager.py",
)
IMPORT_SHADOW_SCRIPTS = (
    "data_collect_chess_commander.py",
    "data_collect_chess_robot.py",
    "eva_chess_commander.py",
    "eva_chess_robot.py",
    "llm_chess_commander.py",
    "hri_chess_commander.py",
    "hri_chess_robot.py",
)


class HardwareProfileTests(unittest.TestCase):
    def load_profile(self, robot):
        with (HARDWARE_DIR / f"{robot}.yaml").open() as stream:
            return yaml.safe_load(stream)

    def test_profiles_use_expected_names(self):
        for robot, ip in (
            ("panda", "192.168.3.111"),
            ("fr3", "192.168.2.200"),
        ):
            with self.subTest(robot=robot):
                profile = self.load_profile(robot)
                self.assertEqual(profile["robot"], robot)
                self.assertEqual(profile["robot_ip"], ip)
                self.assertEqual(profile["move_group"], f"{robot}_arm")
                self.assertEqual(
                    profile["manipulator_group"], f"{robot}_manipulator"
                )
                self.assertEqual(
                    profile["end_effector_link"], f"{robot}_link8"
                )
                self.assertEqual(
                    profile["rviz_config"], f"launch/moveit_{robot}.rviz"
                )

    def test_hardware_launches_default_to_panda(self):
        for filename in HARDWARE_LAUNCHES:
            with self.subTest(filename=filename):
                root = ET.parse(ROOT / "launch" / filename).getroot()
                args = {
                    element.attrib["name"]: element.attrib
                    for element in root.findall("arg")
                }
                self.assertEqual(args["robot"]["default"], "panda")
                includes = [
                    element.attrib["file"] for element in root.findall("include")
                ]
                self.assertIn(
                    "$(find open_chess_robot)/launch/includes/franka_moveit.launch",
                    includes,
                )

    def test_shared_launch_loads_profile_and_matching_rviz(self):
        launch_path = ROOT / "launch" / "includes" / "franka_moveit.launch"
        text = launch_path.read_text()
        self.assertIn(
            "config/hardware/$(arg robot).yaml",
            text,
        )
        self.assertIn(
            "launch/moveit_$(arg robot).rviz",
            text,
        )
        self.assertIn('name="rviz_config"', text)
        self.assertIn('name="arm_id" value="$(arg robot)"', text)


class RuntimePathTests(unittest.TestCase):
    def test_runtime_package_resolves_repository_resources(self):
        self.assertEqual(package_root(), ROOT)
        self.assertEqual(
            resource_path("config", "hardware", "panda.yaml"),
            HARDWARE_DIR / "panda.yaml",
        )

    def test_runtime_scripts_use_portable_python_shebang(self):
        for filename in RUNTIME_SCRIPTS:
            with self.subTest(filename=filename):
                first_line = (ROOT / "scripts" / filename).read_text().splitlines()[0]
                self.assertEqual(first_line, "#!/usr/bin/env python3")

    def test_runtime_scripts_do_not_contain_checkout_path(self):
        for filename in RUNTIME_SCRIPTS:
            with self.subTest(filename=filename):
                text = (ROOT / "scripts" / filename).read_text()
                self.assertNotIn("/home/charles/", text)

    def test_runtime_recognition_uses_chessrec(self):
        commander = (ROOT / "scripts" / "hri_chess_commander.py").read_text()
        self.assertIn("from chessrec.", commander)
        self.assertNotIn("chess_clf", commander)

    def test_scripts_import_source_modules_before_catkin_relays(self):
        for filename in IMPORT_SHADOW_SCRIPTS:
            with self.subTest(filename=filename):
                text = (ROOT / "scripts" / filename).read_text()
                self.assertIn("prefer_source_scripts(__file__)", text)


class RosInterfaceTests(unittest.TestCase):
    def test_recognition_service_is_generated(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        service = (ROOT / "srv" / "RecognizeBoard.srv").read_text()

        self.assertIn("add_service_files", cmake)
        self.assertIn("RecognizeBoard.srv", cmake)
        self.assertIn("string camera_pose", service)
        self.assertIn("bool refresh_geometry", service)
        self.assertIn("string next_turn", service)
        self.assertIn("string camera_side", service)
        self.assertIn("string augment", service)
        self.assertNotIn("use_two_step", service)
        self.assertIn("bool success", service)
        self.assertIn("uint8 STATUS_AMBIGUOUS=1", service)
        self.assertIn("string fen", service)
        self.assertIn("float32 confidence", service)
        self.assertIn("string[] ambiguous_squares", service)
        self.assertIn("string[] debug_image_paths", service)

    def test_recognition_service_runtime_wiring_exists(self):
        robot = (ROOT / "scripts" / "hri_chess_robot.py").read_text()
        commander = (ROOT / "scripts" / "hri_chess_commander.py").read_text()

        self.assertIn("from open_chess_robot.srv import RecognizeBoard", robot)
        self.assertIn("rospy.Service(service_name, RecognizeBoard", robot)
        self.assertIn("BoardRecognitionEngine", robot)
        self.assertIn("from open_chess_robot.srv import RecognizeBoard", commander)
        self.assertIn("rospy.ServiceProxy", commander)
        self.assertIn('self.recognition_backend == "service"', commander)


if __name__ == "__main__":
    unittest.main()
