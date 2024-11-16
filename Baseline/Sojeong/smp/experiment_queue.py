import subprocess
import os

# 실행할 실험 스크립트 목록
experiments = [
    "train4.sh",
    "train3.sh",
    # Add more scripts as needed
]

# 로그 저장 경로
log_dir = "/data/ephemeral/home/Sojeong/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Sojeong/smp/logs/experiment_logs"
os.makedirs(log_dir, exist_ok=True)

# 각 실험 순차적으로 실행
for i, script in enumerate(experiments, start=1):
    print(f"Starting Experiment {i}: {script}")
    
    # 로그 파일 생성
    log_file = os.path.join(log_dir, f"{script.split('.')[0]}_log.txt")
    
    try:
        # 스크립트 실행
        with open(log_file, "w") as logfile:
            result = subprocess.run(
                ["bash", script], 
                stdout=logfile, 
                stderr=subprocess.STDOUT
            )
        
        # 실행 성공 여부 확인
        if result.returncode == 0:
            print(f"Experiment {i} ({script}) completed successfully.")
        else:
            print(f"Experiment {i} ({script}) failed. Check the log: {log_file}")
            break  # 중단: 실패 시 다음 실험 실행 중단

    except Exception as e:
        print(f"An error occurred while running {script}: {e}")
        break  # 중단: 오류 발생 시 다음 실험 실행 중단

print("All queued experiments completed.")

# python experiment_queue.py