
config = {}
config['hand_detection'] = {
    'frozen_pb_path': '../__backup/hand_detection/B/exported_graphs_40185/frozen_inference_graph.pb',
    'th1': 0.25,
    'th2': 0.50
}
config['key_points_estimation'] = {
    'ckpt_path': '../__backup/key_points_estimation/__train/model-29601'
}
config['pose_estimation'] = {
    'ckpt_path': '../__backup/pose_estimation/__train/model-78001'
}
