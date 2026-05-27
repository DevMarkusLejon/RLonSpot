# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path
from threading import Thread

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
)
from spot.mock_spot import MockSpot
from spot.spot import Spot
from utils.event_divider import EventDivider

def start_keyboard_toggle_listener(context: OnnxControllerContext):
    def listen_for_enter():
        while True:
            input()
            context.command_toggle_event.set()
    keyboard_thread = Thread(target=listen_for_enter, daemon=True)
    keyboard_thread.start()

def wait_for_command_toggle(context: OnnxControllerContext, message: str):
    print(message)
    context.command_toggle_event.clear()
    context.command_toggle_event.wait()

def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument("--gamepad-config", type=Path)
    options = parser.parse_args()

    conf_file = orbit.orbit_configuration.detect_config_file(options.policy_file_path)
    policy_file = orbit.orbit_configuration.detect_policy_file(options.policy_file_path)

    context = OnnxControllerContext()
    start_keyboard_toggle_listener(context)
    config = orbit.orbit_configuration.load_configuration(conf_file)
    print(config)

    state_handler = StateHandler(context)
    print("Verbose: ", options.verbose)
    command_generator = OnnxCommandGenerator(context, config, policy_file, options.verbose)

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
        print("Starting Mock Spot")
        spot = MockSpot()
    else:
        print("Starting Real Spot")
        spot = Spot(options)

    with spot.lease_keep_alive():
        try:
            spot.power_on()
            spot.stand(0.0)
            spot.start_state_stream(state_handler)
            
            context.event.clear()
            context.event.wait()
            print("Joint state init post printed in spot_rl_demo")
            print(list(context.latest_state.joint_states.position))

            wait_for_command_toggle(context, "Press Enter or Triangle to begin.")
            # print("Press Enter to begin, Press Enter again to stop")
            # input()
            spot.start_command_stream(command_generator, timeing_policy)
            #input()
            wait_for_command_toggle(context, "Press Enter or Triangle again to stop.")

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
            command_generator.close_log_file()
            print("closed log file")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
