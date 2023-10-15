import numpy as np

from alpha_zero.replay_buffer import ReplayBuffer
from alpha_zero.replay_buffer_experience import ReplayBufferExperience


def test_loading_saving(mocker):
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.makedirs')
    mocker.patch('alpha_zero.replay_buffer.json.dump')
    mocker.patch('alpha_zero.replay_buffer.json.load')

    replay_buffer = ReplayBuffer(
        name_for_saving="test")
    replay_buffer.add_experience(
        ReplayBufferExperience(
            neural_network_input=np.array([[1, 2], [3, 4]]),
            search_probabilities=np.array([[1, 2], [3, 4]]),
            outcome=1
        )
    )
    replay_buffer.add_experience(
        ReplayBufferExperience(
            neural_network_input=np.array([[1, 2, 5, 6], [3, 4, 9, 9]]),
            search_probabilities=np.array([[1, 2, 7, 8], [3, 4, 1, 1]]),
            outcome=-1
        )
    )
    experiences_before = replay_buffer.experiences
    replay_buffer.save_to_file(version=0)
    replay_buffer.reset()
    replay_buffer.load_from_file(version=0)
    for experience_1, experience_2 in zip(experiences_before, replay_buffer.experiences):
        assert experience_1 == experience_2
