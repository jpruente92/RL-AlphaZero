from controlling.controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    controller.train_alpha_zero(
        name_for_saving="Connect_4",
        start_version=1
    )
