import argparse as ag
import json

def get_parser_with_args(metadata_json='\\mnt\\g\\GitHub\\MY-CD-program\\Siam-NestedUNet++-master\\utils\\metadata.json'.replace("\\","/")):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:   # 打开超参数
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata

    return None
