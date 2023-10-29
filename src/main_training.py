from controlling.controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    controller.train_alpha_zero(
        start_version=1,
        parallel=True
    )
