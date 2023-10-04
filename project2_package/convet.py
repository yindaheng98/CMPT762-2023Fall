import re
def convert_state_dict(state_dict):
    converted = {}
    for k in state_dict:
        ck = k
        ck = re.sub("conv2_x", "resblock1", ck)
        ck = re.sub("conv3_x", "resblock2", ck)
        ck = re.sub("conv4_x", "resblock3", ck)
        ck = re.sub("conv5_x", "resblock4", ck)
        ck = re.sub("avg_pool", "avgpool", ck)
        ck = re.sub("residual_function", "residual", ck)
        ck = re.sub("shortcut", "residual_fit", ck)
        converted[ck] = state_dict[k]
    return converted