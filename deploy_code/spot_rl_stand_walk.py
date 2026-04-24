# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

import bosdyn.client.util
import orbit.orbit_configuration
from hid.gamepad import (
    Gamepad,
    GamepadConfig,
    joystick_connected,
    load_gamepad_configuration,
)
from orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
    OnnxStandWalkCommandGenerator
)
from spot.mock_spot import MockSpot
from spot.spot import Spot
from utils.event_divider import EventDivider


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--walk-policy", type=Path)
    parser.add_argument("--stand-policy", type=Path)
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument("--gamepad-config", type=Path)
    options = parser.parse_args()

    context = OnnxControllerContext()
    walk_conf_file = orbit.orbit_configuration.detect_config_file(options.walk_policy)
    walk_policy_file = orbit.orbit_configuration.detect_policy_file(options.walk_policy)
    walk_config = orbit.orbit_configuration.load_configuration(walk_conf_file)

    stand_conf_file = orbit.orbit_configuration.detect_config_file(options.stand_policy)
    stand_policy_file = orbit.orbit_configuration.detect_policy_file(options.stand_policy)
    stand_config = orbit.orbit_configuration.load_configuration(stand_conf_file)

    print("[INFO] BODY CONFIG: ", walk_config)
    print("[INFO] Arm config: ", stand_config)

    state_handler = StateHandler(context)
    print(options.verbose)
    #TODO: Change the class?
    command_generator = OnnxStandWalkCommandGenerator(context, walk_config, stand_config, walk_policy_file, stand_policy_file, options.verbose)

    # 333 Hz state update / 6 => ~56 Hz control updates
    timeing_policy = EventDivider(context.event, 6)

    gamepad = None
    if joystick_connected():
        if options.gamepad_config is not None:
            print("[INFO] loading gamepad config from file")
            gamepad_config = load_gamepad_configuration(options.gamepad_config)
        else:
            print("[INFO] using default gamepad configuration")
            gamepad_config = GamepadConfig()

        gamepad = Gamepad(context, gamepad_config)
        gamepad.start_listening()

    if options.mock:
        spot = MockSpot()
    else:
        spot = Spot(options)

    with spot.lease_keep_alive():
        try:
            spot.power_on()
            spot.stand(0.0)
            spot.start_state_stream(state_handler)

            input()
            spot.start_command_stream(command_generator, timeing_policy)
            input()

        except KeyboardInterrupt:
            print("killed with ctrl-c")

        finally:
            print("stop command stream")
            spot.stop_command_stream()
            print("stop state stream")
            spot.stop_state_stream()
            print("stop game pad")
            if gamepad is not None:
                gamepad.stop_listening()
            print("all stopped")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
