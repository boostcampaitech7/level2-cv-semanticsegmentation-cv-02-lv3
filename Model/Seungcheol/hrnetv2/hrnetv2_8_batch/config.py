# config.py 수정 예시

class Config:
    class MODEL:
        PRETRAINED = 'pth file_path'  # 필요시 경로 추가
        EXTRA = {
            'STAGE1': {
                'NUM_CHANNELS': [64],
                'BLOCK': 'BOTTLENECK',
                'NUM_BLOCKS': [4]
            },
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [48, 96],  # 기존 설정에 맞춰 수정
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192],  # 기존 설정에 맞춰 수정
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192, 384],  # 기존 설정에 맞춰 수정
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM'
            },
            'FINAL_CONV_KERNEL': 1  # 최종 출력 컨볼루션 커널 크기
        }
    
    class DATASET:
        NUM_CLASSES = 29  # 예: 클래스 수

cfg = Config()
