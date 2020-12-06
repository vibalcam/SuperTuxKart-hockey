import numpy as np
import torch
import torchvision
from PIL import Image

from agent.models import load_model

ALL_PLAYERS = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
               'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
# Filtered some of the players that are too big and do not give good images
ALL_PLAYERS_FILTERED = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'hexley', 'kiki', 'konqi', 'nolok',
                        'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
GOAL_POS = np.float32([[0, 75], [0, -75]])  # (0 and 2 coor) Blue, Red

# Steps duration of lost status
LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 40
LAST_PUCK_DURATION = 4
# ALPHA_STEER = 0.95

# True if testing to use HACK_DICT
HACK_ON = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def norm(vector):
    # return np.sqrt(np.sum(np.square(vector)))
    return np.linalg.norm(vector)


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
    # ideas: find best kart
    kart = "wilbert"

    # Load model
    model = load_model('det_final.th').to(device)
    # Resize image to 128x128 and transform to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                torchvision.transforms.ToTensor()])

    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        # For training select player at random
        # self.kart = ALL_PLAYERS_FILTERED[np.random.choice(len(ALL_PLAYERS_FILTERED))]

        # Player info variables
        self.player_id = player_id
        self.team = player_id % 2
        self.position = player_id // 2

        # Timing and status variables
        self.initialize_status()

        print(f"{self.kart}, {device}, team:{self.team}")

    def initialize_status(self):
        # Timing variables
        self.step = 0
        self.timer = 0

        # Status variables
        self.puck_last_pos = 0
        self.step_lost = 0
        self.step_back = 0
        self.normal = True
        self.lost_cooldown = 0
        # self.last_steer = 0

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """

        # Get positions
        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]

        # Reset variables status if scored a goal
        if norm(player_info.kart.velocity) < 1:
            if self.timer == 0:
                self.timer = self.step
            elif self.step - self.timer > 20:
                self.initialize_status()
        else:
            self.timer = 0
        # cur_time = time.perf_counter()
        # if norm(player_info.kart.velocity) < 1:
        #     if self.timer == 0:
        #         self.timer = cur_time
        #     elif cur_time - self.timer > 3:
        #         self.initialize_status()
        # else:
        #      self.timer = 0

        # Get real puck position
        # puck = np.float32(HACK_DICT['state'].soccer.ball.location)[[0, 2]]
        # u = front - kart
        # u = u / np.linalg.norm(u)
        #
        # v = puck - kart
        # v = v / np.linalg.norm(v)
        #
        # theta_puck = np.arccos(np.dot(u, v))
        # signed_theta_puck_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_puck)

        # Get predicted puck position
        img = self.transform(Image.fromarray(image)).to(device)
        pred = self.model.detect(img, max_pool_ks=7, min_score=0.2, max_det=15)  # (score, cx, cy, w, h)
        puck_visible = len(pred) > 0
        if puck_visible:
            # if self.step_lost + 1 == self.step:
            #     puck_pos = np.mean([cx[1] for cx in pred]) * ALPHA_PUCK_POS + self.puck_last_pos * (1 - ALPHA_PUCK_POS)
            # else:
            #     puck_pos = np.mean([cx[1] for cx in pred])

            # Average predictions
            puck_pos = np.mean([cx[1] for cx in pred])
            puck_pos = puck_pos / 64 - 1  # [0, 128] -> [-1, 1]
            puck_size = np.mean([cx[2] for cx in pred])
            puck_size = puck_size / 128  # [0, 128] -> [0, 1]

            # If vary large change, ignore this step
            if self.normal and np.abs(puck_pos - self.puck_last_pos) > 0.5:
                puck_pos = self.puck_last_pos
                self.normal = False
            else:
                self.normal = True

            # Update status variables
            self.puck_last_pos = puck_pos
            self.step_lost = self.step

            # To show when testing
            if HACK_ON and self.position == 0:
                from tournament.utils import HACK_DICT
                HACK_DICT['id'] = self.player_id
                HACK_DICT['predicted'] = (puck_pos, (pred[0])[2] / 64 - 1)
                HACK_DICT['predicted_width'] = puck_size
        elif self.step - self.step_lost < LAST_PUCK_DURATION:
            self.normal = False
            puck_pos = self.puck_last_pos
        else:
            puck_pos = None
            self.step_back = LOST_STATUS_STEPS

        # Opposite goal theta
        u = front - kart
        u = u / np.linalg.norm(u)
        v = GOAL_POS[self.team] - kart
        dist_opp_goal = norm(v)
        v = v / np.linalg.norm(v)

        theta_goal = np.arccos(np.dot(u, v))
        signed_theta_opp_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

        # Self goal theta
        v = GOAL_POS[self.team - 1] - kart
        dist_own_goal = norm(v)
        v = v / np.linalg.norm(v)

        theta_goal = np.arccos(np.dot(u, v))
        signed_theta_self_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

        # ideas: if closer to goal, more important to have angle of goal
        # todo ideas: width and height can be used to know how close is the puck
        dist_opp_goal = (np.clip(dist_opp_goal, 10, 100) - 10) / 90  # [0, 1]
        if self.step_back == 0 and (self.lost_cooldown == 0 or puck_visible):
            if np.abs(signed_theta_opp_goal_deg) < 90:
                aim_point = puck_pos + np.sign(puck_pos - signed_theta_opp_goal_deg / 100) * 0.4 * (1 - dist_opp_goal)
            else:
                aim_point = puck_pos
            # print(f"{aim_point}, {puck_pos}")
            # aim_point = puck_pos
            if self.step_lost == self.step:
                # If have vision of the puck
                acceleration = 0.75
                brake = False
            else:
                # If no vision of the puck
                acceleration = 0
                brake = False
        elif self.lost_cooldown > 0:
            # If already in own goal, start going towards opposite goal
            aim_point = signed_theta_opp_goal_deg / 100
            acceleration = 0.5
            brake = False
            self.lost_cooldown -= 1
        else:
            # If in lost status, back towards own goal
            if dist_own_goal > 10:
                aim_point = signed_theta_self_goal_deg / 100  # [0, 1] aprox
                acceleration = 0
                brake = True
                self.step_back -= 1
            else:
                self.lost_cooldown = LOST_COOLDOWN_STEPS
                self.step_back = 0
                aim_point = signed_theta_opp_goal_deg / 100
                acceleration = 0.5
                brake = False

        if self.position == 1 and self.step < 25:
            # If second car, wait more until start
            acceleration = 0
            brake = False
            # If second car, act as goalie
            # if dist_own_goal > 80:
            #     self.step_back = 15
        # else:
        #     acceleration = 1 if self.step < START_STEPS else acceleration

        # Steer and drift
        steer = np.clip(aim_point * 5, -1, 1)
        # steer = np.clip(aim_point * 5, -1, 1) * ALPHA_STEER + self.last_steer * (1 - ALPHA_STEER)
        # self.last_steer = steer
        drift = np.abs(aim_point) > 0.2
        self.step += 1

        # print(f"{acceleration}, {aim_point}, {steer}, {puck_visible}")

        return {
            'steer': signed_theta_opp_goal_deg if self.step < 25 else steer,
            'acceleration': 1 if self.step < START_STEPS else acceleration,
            'brake': brake,
            'drift': drift,
            'nitro': False, 'rescue': False
        }
