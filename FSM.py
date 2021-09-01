import numpy as np
import pandas as pd
import asyncio
import inspect
import time
from statemachine import StateMachine, State
import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes
import cv2
from cozmo.song import NoteTypes, NoteDurations, SongNote
from abc import ABC, abstractmethod
from pygame import mixer

#try:
    #import winsound
#except ImportError:
    #import os
    #def Beep(frequency,duration):
        #apt-get install beep
        #os.system('beep -f %s -l %s' % (frequency,duration))
#else:
    #def Beep(frequency,duration):
        #winsound.Beep(frequency,duration)

change_state_song = [
    SongNote(noteType=NoteTypes.C2, noteDuration=NoteDurations.Quarter),
    SongNote(noteType=NoteTypes.A2, noteDuration=NoteDurations.Quarter),
    SongNote(noteType=NoteTypes.F2, noteDuration=NoteDurations.Quarter),
]

try:
    from PIL import ImageDraw, ImageFont, Image
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')
def nothing(x):
    pass

def setup(robot: cozmo.robot.Robot):
    robot.move_head(0.2)

class RobotModel():
    def __init__(self, robot: cozmo.robot.Robot):
        self.approach_factor = 0.30
        self.speed =  35
        self.degees = 15
        self.degees_opposite = -15
        self.robot = robot
        self.width_of_view = 256
        self.number_of_runs = 0
        self.stopping_distance = 100
        self.custom_object_index = 0
        self.detected_object = None
        
        #initializing for robot position and object position 
        self.robot_x_position = robot.pose.position.x
        self.robot_y_position = robot.pose.position.y
        self.robot_angle_radian = robot.pose.rotation.angle_z.radians
        self.observed_object = None
        # self.object_x_position = cozmo.util.Pose.position.x
        # self.object_y_position = cozmo.util.Pose.position.y
        # self.object_angle_radian = cozmo.util.Pose.rotation.angle_z.radians
        self.count = 0
        #initializing for searching and detecting cube
        self.detection = False
        self.object_last_position = None
        self.object_last_degree = None
        self.custom_object_world = robot.world
        self.custom_object = None
        self.custom_objects_reached = []
        self.marked = None

    # works. don't change it (note to myself)
    async def show_state_on_face(self, state_name):
        image = Image.new('1', cozmo.oled_face.dimensions(), (1))
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), state_name, font=font)
        # resize to fit on Cozmo's face screen
        resized_image = image.resize(cozmo.oled_face.dimensions(), resample=0)
        # convert the image to the format used by the oled screen
        face_image = cozmo.oled_face.convert_image_to_screen_data(resized_image,
                                                                 invert_image=True)        
        await self.robot.display_oled_face_image(
            screen_data=face_image,
            duration_ms=1000,
            in_parallel=True).wait_for_completed()

    def getRobotPosition(self):
        return self.object_last_position
    


    async def more_targets_exist(self):
        return len(self.custom_objects_reached) < len(self.robot.world.custom_objects)
        
    async def target_custom_object_is_close(self):
        distance_to_object = self.get_distance_to_object(self.observed_object)
        is_close = distance_to_object < self.stopping_distance
        print("is close", is_close, distance_to_object," "*10, end="\r")
        return is_close
        
    def go_to_position(self, position, y_adjust = 1, turn_only = False):
        
        object_position_relative_to_robot = transformation(
            self.robot.pose.position.x,
            self.robot.pose.position.y,
            self.robot.pose.rotation.angle_z.radians,
            position.x, 
            position.y
        )
        
        distance_to_object = object_position_relative_to_robot[0]
        object_relative_y = object_position_relative_to_robot[1]
        if y_adjust > 0:
            object_relative_y += np.sign(object_relative_y) * y_adjust
        
        turn_radius = 10
        omega = np.arctan(object_relative_y / distance_to_object)
        rght_angular_velocity = omega * (turn_radius + distance_to_object / 2) * 0.6
        left_angular_velocity = omega * (turn_radius - distance_to_object / 2) * 0.6
        # rght_angular_velocity = -object_relative_y
        # left_angular_velocity = object_relative_y
        
        r_wheel_speed = (self.speed if not turn_only else 0) + rght_angular_velocity
        l_wheel_speed = (self.speed if not turn_only else 0) + left_angular_velocity
        
        if abs(distance_to_object < self.stopping_distance):
            # robot_state_machine.stop_fully
            pass
        else:
            duration = (distance_to_object * self.approach_factor) / self.speed
            self.robot.drive_wheel_motors(
                l_wheel_speed, 
                r_wheel_speed)
                    
    def get_distance_to_object(self, obj):
        x,y = (
            obj.pose.position.x,
            obj.pose.position.y
        )
        relative_object_pose = transformation(
            self.robot.pose.position.x,
            self.robot.pose.position.y,
            self.robot.pose.rotation.angle_z.radians,
            x,
            y
        )
        return relative_object_pose[0]
        
    async def target_custom_object_observed(self):
        return self.detection
    
    async def target_custom_object_lost(self):
        return self.detection == False and self.object_last_position is not None

    async def target_custom_object_observed_moved(self):
        return self.detection == True and self.object_last_position.x > 100
    
def transformation(
    robot_xG, 
    robot_yG, 
    robot_angle_radian, 
    object_xG, 
    object_yG
    ):
    
    #Transformation of an object relative to the robot
    trans = np.array([
        [np.cos(robot_angle_radian), -1*np.sin(robot_angle_radian), robot_xG],
        [np.sin(robot_angle_radian), np.cos(robot_angle_radian), robot_yG],
        [0, 0 ,1]
        ])

    inverse_of_trans = np.linalg.pinv(trans)
    object_global_p = [object_xG, object_yG, 1]
    result = np.matmul(inverse_of_trans, object_global_p)

    # should be something like [x,y,1] 
    return result

class state_behavior_base():
    def __init__(self, name=''):
        pass
  

class StateObserver(ABC):
    @abstractmethod
    async def update(self, robot: cozmo.robot.Robot, cubePose: cozmo.util.Pose):
        pass

class AbstractAsyncState(ABC, State):
    @abstractmethod
    async def perform(self, robot: cozmo.robot.Robot, cubePose: cozmo.util.Pose):
        pass
    
class search_for_object_state(State):
    #Searches for object(detects object) and faces it
    async def state_machine(robot_model: RobotModel):
        count = 0
        if robot_model.custom_object is None:
            #If you dont see any object keep turning 
            robot_model.robot.drive_wheel_motors(
                5, 
                -5)

            # for robot_model.custom_object in robot_model.custom_object_world.visible_objects:
            #     continue
        #else:
            #half_view_width = (robot_model.width_of_view / 2)
            #count += (robot_model.number_of_runs == np.sign(half_view_width) * -1)

            #if abs(half_view_width) > 30:
                #robot_model.number_of_runs = np.sign(half_view_width)
                #turn_degrees = cozmo.util.degrees(
                    #np.sign(half_view_width) * np.max([half_view_width / 7 - count, 5]))
                #await robot_model.robot.turn_in_place(turn_degrees).wait_for_completed()
            

class approach_the_object_state(State):
    async def state_machine(robot_model: RobotModel):
        object_position_relative_to_robot = None

class approach_the_object_state(State):
    def state_machine(robot_model: RobotModel):
        robot_model.go_to_position(robot_model.object_last_position)

class stop_the_robot_state(State):
    def state_machine(robot_model: RobotModel):
        #stops the wheels for the robot fully
        robot_model.robot.stop_all_motors()
        
class relocate_robot_cube_out_of_view(State):
    def state_machine(robot_model: RobotModel):
        
        count = 0
        if robot_model.custom_object is not None:
            half_view_width = (robot_model.width_of_view / 2)
            print(half_view_width)
            count += (robot_model.number_of_runs == np.sign(half_view_width) * -1)
            print(robot_model.object_last_degree)
            if abs(half_view_width) > 30:
                if robot_model.object_last_degree < -0.60:
                    await robot_model.robot.drive_wheels(
                    15, 
                    -15,
                    l_wheel_acc=None, 
                    r_wheel_acc=None, 
                    duration=2)

        
            
# class finding_object_out_of_view(State):
#     async def state_machine(self, robot_model: RobotModel):
#         #If you see the object turn the oppositeway if it is out of view
#         if robot_model.custom_object is None:
#             await robot.drive_wheels(
#                 -50.0, 
#                 50.0,
#                 l_wheel_acc=None, 
#                 r_wheel_acc=None, 
#                 duration=2).wait_for_completed()
#         else:
#             await robot.drive_wheels(
#                 50.0, 
#                 -50.0,
#                 l_wheel_acc=None, 
#                 r_wheel_acc=None, 
#                 duration=2).wait_for_completed()
            
            
class changing_robot_location(State):
    async def state_machine(self, robot_model: RobotModel):
        # Relocating the robot to  new position
        if self.robot_x_position > 10 and self.robot_y_position > 10:
            await robot.drive_wheels(
                50.0, 
                50.0,
                l_wheel_acc=None, 
                r_wheel_acc=None, 
                duration=2)
        else:
            await robot.drive_wheels(
                50.0, 
                -50.0,
                l_wheel_acc=None, 
                r_wheel_acc=None, 
                duration=2)
                    
        
class object_annotator(cozmo.annotate.Annotator):
    image_of_cube = None
    def apply(self, image, scale):
        dimestion = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)
        
        if object_annotator.image_of_cube is not None:
            
            object_annotator.image_of_cube = np.multiply(object_annotator.image_of_cube, 2)
            
            image_box = cozmo.util.ImageBox(object_annotator.image_of_cube[0]-object_annotator.image_of_cube[2]/2,
                                            object_annotator.image_of_cube[1]-object_annotator.image_of_cube[2]/2,
                                            object_annotator.image_of_cube[2], object_annotator.image_of_cube[2])
            
            cozmo.annotate.add_img_box_to_image(image, image_box, "green", text=None)

            object_annotator.image_of_cube = None
    
    
class RobotStateMachine(StateMachine):
    """This is a state machine for cozmo for going from state to state"""
    
    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model
        self.behaviors = {}
        self.conditions = {}
        super().__init__()
        
    def add_state_behavior(self, state_name, behavior):
        if state_name not in self.behaviors:
            self.behaviors[state_name] = []
        self.behaviors[state_name].append(behavior)
        
    def add_state_condition(self, state_name, condition, transition):
        if state_name not in self.conditions:
            self.conditions[state_name] = []
        self.conditions[state_name].append({"condition":condition,"transition":transition})
        
    async def do_state_behavior(self):
        state = self.current_state
        for b in self.behaviors[state.name]:
            if inspect.iscoroutinefunction(b):
                await b(self.robot_model)
            else:
                b(self.robot_model)
                                
    async def evaluate_conditions(self):
        state = self.current_state
        if state.name in self.conditions:
            for c in self.conditions[state.name]:
                condition = c["condition"]
                transition = c["transition"]
                is_satisfied = False
                if inspect.iscoroutinefunction(condition):
                    is_satisfied = await condition()
                else:
                    is_satisfied = condition()
                if is_satisfied:
                    transition()
                    self.robot_model.robot.stop_all_motors()
                    await self.robot_model.show_state_on_face(self.current_state.name)
                    break
                                
    async def object_is_not_in_view(self, evt, **kw):
        self.robot_model.detection = False
        pass

    async def object_is_in_view(self, evt, **kw):
        #Whenever an object is in view
        # custom_objects = self.robot_model.robot.world.custom_objects
        self.detected_object = evt.obj
        if isinstance(evt.obj, CustomObject) and evt.obj.object_type.id not in self.robot_model.custom_objects_reached:
            # print("object detected", kw["pose"].position.x, kw["pose"].position.y, end="\r")
            self.robot_model.detection = True
            if evt.obj is not None:
                self.robot_model.observed_object = evt.obj
            self.robot_model.object_last_position = kw["pose"].position
            self.robot_model.object_last_degree = kw["pose"].rotation.angle_z.radians
        else:
            print("not a custom object")

    async def make_custom_AR_markers_object(self):
        #Make custom AR marker object
        self.robot_model.robot.add_event_handler(
            cozmo.objects.EvtObjectObserved, 
            self.object_is_in_view)
        self.robot_model.robot.add_event_handler(
            cozmo.objects.EvtObjectDisappeared, 
            self.object_is_not_in_view)
        self.robot_model.robot.add_event_handler(
            cozmo.objects.EvtObjectMoving, 
            self.target_custom_object_moved)
        await self.robot_model.custom_object_world.define_custom_cube(
            CustomObjectTypes.CustomType00,
            CustomObjectMarkers.Diamonds4, 44, 28, 28, True)
        await self.robot_model.custom_object_world.define_custom_cube(
            CustomObjectTypes.CustomType01,
            CustomObjectMarkers.Circles3, 44, 28, 28, True)
    
    #Searches for cube(detects cube) / Turns facing the cube
    spinning = search_for_object_state('spinning', initial=True)
    
    #Relocating the robot to another position since the current position can't find the cube
    relocate_the_cube = relocate_robot_cube_out_of_view('relocate_the_cube')

    #Approaches the cube / Search for a second time to see if that was the right cube
    approaching = approach_the_object_state('approaching')
    
    # Failed to find the cube by spinning, so I'm looking for a new vantage point.
    changing_location = changing_robot_location('changing_location')
    
    # # Object exits the view
    # finding_object = finding_object_out_of_view('finding_object')
    
    #Stop the robot
    stopped = stop_the_robot_state('stopped')

##################################################################################################
    #Search and detects cube while facing the
    #cube and approach the cube while searching a second time
    approach = approaching.from_(
        spinning, 
        changing_location, 
        approaching, 
        relocate_the_cube
    )

    #Approach the cube and will stop once it detects the cube"
    stop = stopped.from_(
        approaching, 
        spinning, 
        #finding_object, 
        relocate_the_cube
    )

    #Stop the robot to search
    search = spinning.from_(
        approaching, 
        stopped, 
        #finding_object, 
        relocate_the_cube, 
        changing_location, 
        spinning
    )
    
    continue_searching = spinning.to.itself()
    
    # Go from spinning around to changing vantage point (maybe not necessary)
    change_location = changing_location.from_(spinning)
    
    #relocate if robot is out of view completely
    # find_object = spinning.to(finding_object)
    
    # Relocate the robot if it is out of center view
    relocate = approaching.to(relocate_the_cube)

    #Stops the robot and can determind if we want to search for something else
    stop_fully = stopped.to.itself()
    
    # on_enter_X gets triggered whenever the SM enters these states.
    # TODO: add some kind of "beep" here to meet lab requirements
    def on_enter_spinning(self):
        self.on_state_change()
    
    def on_enter_changing_location(self):
        self.on_state_change()
   
    def on_enter_approaching(self):
        self.on_state_change()
    
    def on_enter_stopped(self):
        # start looking for the next custom cube in the cycle
        if self.robot_model.custom_object is not None:
            cube_id = self.robot_model.custom_object.object_type.id
            print(f"MARKING CUBE {cube_id} AS REACHED. (WILL NOT APPROACH AGAIN.)")
            self.robot_model.custom_objects_reached.append(cube_id)
        self.on_state_change()
        self.robot_model.custom_object = None
        self.robot_model.detection = False
    
    def on_enter_relocate_the_cube(self):
        self.on_state_change()
        
    # def on_enter_finding_object(self):
    #     self.on_state_change()
        
    def on_state_change(self):
        # an auditory signal for a state change
        self.robot_model.robot.stop_all_motors()
        #winsound.Beep(int(880), 250)
        mixer.init() 
        sound=mixer.Sound("bell.wav")
        sound.play()
        # print statement on your terminal to show state changes
        # self.robot_model.robot.play_song(change_state_song)
        print(f'ENTERED STATE \'{self.current_state.name}\'')
        
class StateShower(object):
    def __init__(self, machine: RobotStateMachine):
        self.state_to_show = machine
        
    async def update(self, robot: cozmo.robot.Robot):
        print("Current State is %s" % (self.state_to_show.current_state.name))
        
        # if self.state_to_show.robot_model.detection == True:
        #     print("Searching, Spinning for now")
        #     self.state_to_show.approach()
        # elif self.state_to_show.current_state.name == 'search_for_object':
        #     print("Found the Marker", self.state_to_show.current_state.name)
        #     self.state_to_show.approach()


