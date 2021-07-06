from os import listdir
from os.path import join, isdir
import urbs.admm_async.input_output as io

result_dir = 'result/Run-20210706T1251'

dirs = [
    f for f in listdir(result_dir)
    if isdir(join(result_dir, f)) and io.results_path('') in listdir(join(result_dir, f))
]

# files = listdir(result_dir)
# print('files')
# print(files)
# dirs = [f for f in files if isdir(f)]
# print('dirs')
# print(dirs)
# for f in dirs:
#     print(f)
#     print(listdir(f))
#     print(io.results_path())
#     print(io.results_path() in listdir(f))

stats = []

for name in dirs:
    results = io.load_results(join(result_dir, name))
    time = results['admm_time']
    gap = results['objective_gap']

    stats.append({
        'name': name,
        'time': time,
        'gap': 100 * abs(gap),
    })

io.save_overview(stats, result_dir)