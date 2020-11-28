import time

import numpy as np

from model.models import load_model
from tournament.utils import HACK_DICT


ALL_PLAYERS = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
                       'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']

GOAL_POS = np.float32([[0, 75], [0,  -75]])    # (0 and 2 coor) Blue, Red


def norm(vector):
    return np.sqrt(np.sum(np.square(vector)))


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.player_id = player_id
        self.kart = ALL_PLAYERS[np.random.choice(len(ALL_PLAYERS))]
        self.team = player_id % 2
        self.position = player_id // 2
        self.time = time.perf_counter()
        self.time_stop = 0
        self.time_back = 0

        # Load model
        self.model = load_model()
        self.model.eval()

        print(self.kart)
        
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        # Get positions
        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]

        # Get real puck position
        puck = np.float32(HACK_DICT['state'].soccer.ball.location)[[0, 2]]
        u = front - kart
        u = u / np.linalg.norm(u)

        v = puck - kart
        v = v / np.linalg.norm(v)

        theta_puck = np.arccos(np.dot(u, v))
        signed_theta_puck_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_puck)

        # Get predicted puck position
        # todo
        # pred = self.model.detect(image, max_det=1)[0]   # (score, cx, cy, w, h)
        # puck_pos = pred[1]
        # puck_size = pred[2]

        # Opposite goal theta
        v = GOAL_POS[self.team] - kart
        v = v / np.linalg.norm(v)

        theta_goal = np.arccos(np.dot(u, v))
        signed_theta_opp_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

        # Self goal theta
        v = GOAL_POS[self.team - 1] - kart
        v = v / np.linalg.norm(v)

        theta_goal = np.arccos(np.dot(u, v))
        signed_theta_self_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

        # todo ideas: if closer to goal, more important to have angle of goal
        # todo ideas: width and height can be used to know how close is the puck
        theta_degrees = signed_theta_puck_deg + np.sign(signed_theta_puck_deg - signed_theta_opp_goal_deg) * 1
        cur_time = time.perf_counter()

        action = {
            'steer':  0,
            'acceleration': 1 if cur_time - self.time < 15 else 0.75,
            'brake': 0,
            'drift': 0,
            'nitro': False, 'rescue': False
        }

        # If does not have vision of the puck return to own goal
        # if np.abs(np.degrees(signed_theta_self_goal) - np.degrees(signed_theta_puck)) < 90:
        #     theta = signed_theta_self_goal
        #     action['acceleration'] = 0
        #     action['brake'] = True

        # Timings wall
        if norm(np.float32(player_info.kart.velocity)[[0, 2]]) < 5:
            if self.time_stop == 0:
                self.time_stop = cur_time
            elif cur_time - self.time_stop > 4:
                self.time_back = cur_time
        else:
            self.time_stop = 0

        if cur_time - self.time_back > 2:
            self.time_back = 0

        if self.time_back > 0:
            theta_degrees = signed_theta_self_goal_deg
            action['acceleration'] = 0
            action['brake'] = True

        # If second car, wait more until start
        if self.position == 0:
            if cur_time - self.time < 10:
                action['acceleration'] = 0
                action['brake'] = 0


        # Steer and drift
        action['steer'] = theta_degrees / 15
        action['drift'] = np.abs(theta_degrees) > 20

        return action

        # return {
        #     'steer': 20 * signed_theta_puck,
        #     'acceleration': 0.5,
        #     'brake': False,
        #     'drift': np.degrees(theta_puck) > 60,
        #     'nitro': False, 'rescue': False}