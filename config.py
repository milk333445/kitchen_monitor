import yaml

def load_config(file_path = './setting/settings.yaml'):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    config = load_config()
    bbox_color = tuple(config['bbox_color'])
    bbox_thickness = config['bbox_thickness']
    bbox_labelstr = config['bbox_labelstr']
    kpt_color_map = config['kpt_color_map']
    kpt_labelstr = config['kpt_labelstr']
    skeleton_map = config['skeleton_map']
    line = config['cooking_roi']
    loading_bar_color = tuple(config['loading_bar_color'])
    loading_bar_radious = tuple(config['loading_bar_radious'])
    loading_bar_labelstr = config['loading_bar_labelstr']
    pre_state = config['pre_state']
    high_confidence_count = config['high_confidence_count']
    difference_count = config['difference_count']
    inside_and_interact_count = config['inside_and_interact_count']
    
    print(pre_state)
    
    
    
    