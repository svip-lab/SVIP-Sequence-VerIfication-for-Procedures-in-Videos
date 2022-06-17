
def frames_preprocess(frames):

    bs, c, h, w, num_clip = frames.size()
    frames = frames.permute(0, 4, 1, 2, 3)
    frames = frames.reshape(-1, c, h, w)

    return frames