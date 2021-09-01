import asyncio
import sys
import cv2
import numpy as np
import cozmo
import threading
import time
import os
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes
from FSM import *
from glob import glob
from events import *
from typing import List

state_observers: List[StateObserver] = []

def attach(observer: StateObserver):
    state_observers.append(observer)

async def run(robot: cozmo.robot.Robot):
    robot.abort_all_actions(False)

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('image box', object_annotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure = True

    gain,exposure = 3.7,6 
    mode = 0
    
######################################
    # fs_machine = RobotModel(robot)
    model = RobotModel(robot)
    robot_state_machine = RobotStateMachine(model)
    await robot_state_machine.make_custom_AR_markers_object()
    
    robot_state_machine.add_state_behavior(
        'spinning', 
        search_for_object_state.state_machine)
    
    robot_state_machine.add_state_behavior(
        'stopped', 
        stop_the_robot_state.state_machine)
    
    robot_state_machine.add_state_behavior(
        'changing_location',
        changing_robot_location.state_machine)
    
    robot_state_machine.add_state_behavior(
        'relocate_the_cube', 
        relocate_robot_cube_out_of_view.state_machine)
    
    # robot_state_machine.add_state_behavior(
    #     'finding_object', 
    #     finding_object_out_of_view.state_machine)
    
    robot_state_machine.add_state_behavior(
        'approaching', 
        approach_the_object_state.state_machine)
    
    robot_state_machine.add_state_condition(
        state_name='spinning',
        condition=model.target_custom_object_observed,
        transition=robot_state_machine.approach
    )
    
    #####
    # robot_state_machine.add_state_condition(
    #     state_name='approaching',
    #     condition=model.target_custom_object_observed_moved,
    #     transition=robot_state_machine.relocate
    # )
    
    # robot_state_machine.add_state_condition(
    #     state_name='spinning',
    #     condition=model.target_custom_object_observed_moved,
    #     transition=robot_state_machine.relocate
    # )
    
    #####
    robot_state_machine.add_state_condition(
        state_name='finding_object',
        condition=model.target_custom_object_observed,
        transition=robot_state_machine.approach
    )
    
    # robot_state_machine.add_state_condition(
    #     state_name='spinning',
    #     condition=model.target_custom_object_lost,
    #     transition=robot_state_machine.find_object
    # )
    
    # robot_state_machine.add_state_condition(
    #     state_name='finding_object',
    #     condition=model.target_custom_object_observed_moved,
    #     transition=robot_state_machine.relocate
    # )
    
    robot_state_machine.add_state_condition(
        state_name='relocate_the_cube',
        condition=model.target_custom_object_observed,
        transition=robot_state_machine.approach
    )
    
    robot_state_machine.add_state_condition(
        state_name='approaching',
        condition=model.target_custom_object_is_close,
        transition=robot_state_machine.stop
    )
    
    # robot_state_machine.add_state_condition(
    #     state_name='approaching',
    #     condition=model.target_custom_object_lost,
    #     transition=robot_state_machine.relocate
    # )
    
    robot_state_machine.add_state_condition(
        state_name='stopped',
        condition=model.more_targets_exist,
        transition=robot_state_machine.search
    )
    
    shower = StateShower(robot_state_machine)
    attach(shower)

    try:
        mid_angle = (cozmo.robot.MIN_HEAD_ANGLE.degrees + cozmo.robot.MAX_HEAD_ANGLE.degrees)/2
        mid_angle -= 14
        await robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE, in_parallel=False).wait_for_completed()

        await robot.set_lift_height(0.0, in_parallel=False).wait_for_completed()
        await robot.set_head_angle(cozmo.util.degrees(mid_angle), in_parallel=False).wait_for_completed()

        while True:
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)   #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

                if mode == 1:
                    robot.camera.enable_auto_exposure = True
                else:
                    robot.camera.set_manual_exposure(exposure, gain)
            

            await robot_state_machine.do_state_behavior()
            await robot_state_machine.evaluate_conditions()
            
    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)
