# run_scripts: 0.9v

import pandas as pd
import os 
import subprocess
import datetime
import time
from util import Util
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

server_name = "GPU1"
queue_file_name = "queue.csv"
datetime_format = "%y/%m/%d %H:%M:%S"

queue_fold_path = os.path.dirname(os.path.realpath(__file__))
queue_path = os.path.join(queue_fold_path, queue_file_name)

now = datetime.datetime.now
run_cnt = 0
util = Util(queue_path, "", server_name)
os.environ['MKL_THREADING_LAYER'] = 'GNU'

while True:
    try:
        current, nleft = util.get_current_exec()

        if current is None:
            util.log("큐를 모두 실행했으므로 종료합니다.")
            break

        # 명령 실행 인수 준비
        file_path = os.path.normpath(current['file_path'])
        fold_path, file_name = os.path.split(file_path)
        argument = current['argument']
        run = f"python {file_name} {argument}" 
        
        # 시작 시간 업데이트
        start_time, start_time_str = util.get_current_time()
        current['start'] = start_time_str
        util.update_queue_from_current(current)
        
        # 명령 시작
        util.log(f"실행 시작 - 대기 중: {nleft}")
        util.log(f"{fold_path} {run}")
        subprocess.call(run, cwd = fold_path, shell = True)

        # 실행시간 측정 및 로깅
        end_time, end_time_str = util.get_current_time()
        elapsed = (end_time - start_time).total_seconds() / 3600 # 시 단위로 호나산
        util.log(f"실행 완료 - 실행 시간: {elapsed:.1f} h", end_time_str)

        # 결과 업데이트
        current['end'] = end_time_str
        util.update_queue_from_current(current)
        time.sleep(1) # 오작동 시 메시지 많이 보내는 것 방지

    except:
        util.log("실행 실패")
        current['end'] = end_time_str
        util.update_queue_from_current(current)
        time.sleep(1)
        
