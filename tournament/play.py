from .utils import Player
from .utils import Tournament
from argparse import ArgumentParser
import importlib

import pystk


class DummyPlayer:
    def __init__(self, team=0):
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(
            controller=pystk.PlayerConfig.Controller.AI_CONTROL,
            team=self.team)
    
    def __call__(self, image, player_info):
        return dict()


if __name__ == '__main__':
    parser = ArgumentParser("Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-s', '--save_loc', help="Do you want to record data?")
    # Does not work correctly on windows
    # parser.add_argument('--save_mp4', help="Do you want to record?")
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show_mask', action='store_true')
    parser.add_argument('-f', '--num_frames', default=1000, type=int, help="How many steps should we play for?")
    parser.add_argument('players', nargs='+', help="Add any number of players. List python module names or `AI` for AI players). Teams alternate.")
    args = parser.parse_args()
    
    players = []
    for i, player in enumerate(args.players):
        if player == 'AI':
            players.append(DummyPlayer(i % 2))
        else:
            players.append(Player(importlib.import_module(player).HockeyPlayer(i), i % 2))
        
    tournament = Tournament(players)
    score = tournament.play(save_dir=args.save_loc, max_frames=args.num_frames, show=args.show, save_mp4=None,
                            show_mask=args.show_mask)
    tournament.close()
    print('Final score', score)