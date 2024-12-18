import utils, os
from argparse import ArgumentParser
from train import Trainer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_path', type=str, default=None) 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # main.py 실행 시 conf/default.json 파일에서 기본 실험 조건을 불러옵니다.
    # argument '--exp_path'를 사용하여 기본 조건을 다른 실험 조건 파일로 덮어쓸 수 있습니다.
    # 사용 예시: python main.py --exp_path conf/exp1.json
    # exp1.json 파일에서 변경할 조건(예: train_json_name, valid_json_name)을 지정하여 실험을 수행할 수 있습니다.
    # train.py의 결과로 저장된 weight는 inference.py를 이용하여 train, valid, test 예측 결과를 얻을 수 있습니다. 
    

    # exp_path 불러오기
    args = parse_args() 
    exp_path = args.exp_path

    # 실험조건 불러오기
    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # train.py의 디렉토리
    conf = utils.load_conf(work_dir_path=work_dir_path, rel_exp_path=exp_path) 
    conf['work_dir_path'] = work_dir_path
    conf['run_name'] = conf['run_name_format'].format(**conf)

    # model_dir 및 run_name 겹침 여부 확인 및 수정
    conf['model_dir_path'] = os.path.join(work_dir_path, f"trained_models/{conf['run_name']}")
    conf['model_dir_path'] = utils.renew_if_path_exist(conf['model_dir_path'])
    conf['run_name'] = os.path.basename(conf['model_dir_path'])

    # wandb 설정
    if conf['debug'] == False:
        utils.load_wandb(conf) 

    # 사전에 학습된 결과 폴더가 지정되었으면, weight 폴더 경로로 재지정
    if conf['trained_path'] is not None:
        conf['trained_path'] = utils.get_saved_model_dir_path(conf['trained_path'])

    # 실험 조건 저장
    save_conf_path = os.path.join(conf['model_dir_path'], 'exp.json')
    os.makedirs(conf['model_dir_path'], exist_ok=True)
    utils.save_json(conf, save_conf_path)

    # 학습 시작
    utils.set_seed() # seed 고정
    trainer = Trainer(conf)
    trainer.train()

