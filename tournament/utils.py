import os
import pathlib

import pystk
import numpy as np
from PIL import Image

HACK_DICT = dict()


class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class DataCollector(object):
    def __init__(self, destination):
        self.images = list()
        self.destination = pathlib.Path(destination)

        # Create dirs
        os.makedirs(f"{destination}/images", exist_ok=False)
        os.makedirs(f"{destination}/masks", exist_ok=False)

    def save_frame(self, race, frame, n_players=4):
        for i in range(n_players):
            image = race.render_data[i].image
            Image.fromarray(image).save(f"{self.destination}/images/{i}_{frame}.png")

            mask = race.render_data[i].instance == 134217729
            Image.fromarray(mask).save(f"{self.destination}/masks/{i}_{frame}.png")


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save_dir=None, max_frames=50):
        state = pystk.WorldState()
        if save_dir is not None:
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir, exist_ok=False)
            data_collector = DataCollector(save_dir)

        # To show the agent playing
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()

            list_actions = []
            for i, p in enumerate(self.active_players):
                # HACK_DICT data
                HACK_DICT['race'] = self.k
                HACK_DICT['render_data'] = self.k.render_data[i]
                HACK_DICT['kart'] = state.karts[i]
                HACK_DICT['state'] = state

                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                # if save is not None:
                #     PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))

            # To show the agent playing
            ax.clear()
            ax.imshow(self.k.render_data[0].image)
            # Show player prediction
            if 'predicted' in HACK_DICT:
                pred = HACK_DICT['predicted']       # w, h [-1, 1]
                pred_w = HACK_DICT['predicted_width']        # w [0, 1]
                pred = ((pred[0] + 1) / 2 * self.graphics_config.screen_width, (pred[1] + 1) / 2 * self.graphics_config.screen_height)
                circle = plt.Circle(pred, radius=(pred_w / 2) * self.graphics_config.screen_width, fill=False)
                circle2 = plt.Circle(pred, radius=3, fill=True, color='red')
                ax.add_patch(circle)
                ax.add_patch(circle2)
            plt.pause(1e-3)

            # Save data
            if save_dir is not None:
                data_collector.save_frame(self.k, t, n_players=len(self.active_players))

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        # Save mp4
        # if save is not None:
        #     import subprocess
        #     for i, p in enumerate(self.active_players):
        #         dest = os.path.join(save, 'player%02d' % i)
        #         output = save + '_player%02d.mp4' % i
        #         subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])

        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k
