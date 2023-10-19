from controlling.controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    controller.simulate_tournament(
        no_games=25,
        seconds_per_move=0.1
    )
