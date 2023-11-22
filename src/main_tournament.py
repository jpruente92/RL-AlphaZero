from controlling.controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    controller.simulate_tournament(
        no_games=1,
        seconds_per_move=1,
        with_gui=True
    )
