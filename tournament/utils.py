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
    tmp_loc = 'tmp'

    def __init__(self, destination=tmp_loc):
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


def update_chart(plt, ax1, ax2, tournament, frame):
    id_show = 0 if 'id' not in HACK_DICT else HACK_DICT['id']
    ax1.clear()
    ax1.imshow(tournament.k.render_data[id_show].image)
    # Show player prediction
    if 'predicted' in HACK_DICT:
        pred = HACK_DICT['predicted']  # w, h [-1, 1]
        pred_w = HACK_DICT['predicted_width']  # w [0, 1]
        pred = (
        (pred[0] + 1) / 2 * tournament.graphics_config.screen_width, (pred[1] + 1) / 2 * tournament.graphics_config.screen_height)
        circle = plt.Circle(pred, radius=(pred_w / 2) * tournament.graphics_config.screen_width, fill=False)
        circle2 = plt.Circle(pred, radius=3, fill=True, color='red')
        ax1.add_patch(circle)
        ax1.add_patch(circle2)

    # Show predicted mask
    if ax2 is not None:
        ax2.clear()
        ax2.imshow(HACK_DICT['pred_mask'])
    plt.pause(1e-3)

    # Save fig
    # plt.savefig(f"tmp/{frame}.png")


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

    def play(self, save_dir=None, max_frames=50, show=False, save_mp4=None, show_mask=False):
        state = pystk.WorldState()
        if save_dir is not None:
            data_collector = DataCollector(save_dir)
        if save_mp4 is not None:
            data_collector = DataCollector()

        # To show the agent playing
        if show:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(1, 2 if show_mask else 1)
            ax2 = None
            if show_mask:
                (ax1, ax2) = ax1

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
                HACK_DICT['show_mask'] = show_mask

                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

            # To show the agent playing
            if show:
                update_chart(plt, ax1, ax2, self, t)

            # Save data
            if save_dir is not None or save_mp4 is not None:
                data_collector.save_frame(self.k, t, n_players=len(self.active_players))

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        # Save mp4
        # if save_mp4 is not None:
        #     # Does not work in windows
        #     # import ffmpeg
        #     import shutil
        #     import subprocess
        #     for i, p in enumerate(self.active_players):
        #         # ffmpeg.input(f'/{DataCollector.tmp_loc}/images/{i}_*.png', pattern_type='glob', framerate=25) \
        #         #     .output(f'{save_mp4}.mp4')\
        #         #     .run()
        #         data_loc = f"tmp/images/{i}"
        #         dest = os.path.join(save_mp4, 'player%02d' % i)
        #         output = save_mp4 + '_player%02d.mp4' % i
        #         subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', f'{data_loc}_%05d.png', output])
        #     shutil.rmtree(f'/{DataCollector.tmp_loc}')

        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k
