import sys
sys.path.insert(0, './')
import datetime
import numpy as np
import os
import time
from datetime import timedelta
from scipy.spatial import distance_matrix
from utils import run_all_in_pool
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call, check_output, CalledProcessError
from urllib.parse import urlparse

def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.4.tgz"):

    cwd = os.path.abspath(os.path.join("problems", "vrp", "lkh"))
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)

def solve_lkh_log(executable, directory, name, loc, runs=1, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, name=name)

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsplib(tour_filename)
            # save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "MAX_TRIALS": 1,
        "RUNS": 1,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")

def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour.tolist()


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def lkh_solve(opts, dataset, instance_idx):
    dataset_basename = f'cvrp{opts.problem_size}'
    results_dir = os.path.join(opts.results_dir, "subtsp", dataset_basename)
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, str(instance_idx), str(datetime.datetime.now()))
    
    # assert opts.f or not os.path.isfile(
    #     out_file), "File already exists! Try running with -f option to overwrite."        

    target_dir = os.path.join(results_dir, "{}-{}-{}".format(
        dataset_basename,
        str(instance_idx),
        str(datetime.datetime.now())
    ))
    # assert opts.f or not os.path.isdir(target_dir), \
    #     "Target dir already exists! Try running with -f option to overwrite."

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # TSP contains single loc array rather than tuple
    _dataset = [(instance, ) for instance in dataset]
    use_multiprocessing = False
    executable = get_lkh_executable()

    def run_func(args):
        return solve_lkh_log(executable, *args, runs=1, disable_cache=opts.disable_cache)

    results, parallelism = run_all_in_pool(
        run_func,
        target_dir, _dataset, opts, use_multiprocessing=use_multiprocessing
        )

    costs, tours, durations = zip(*results)

    save_dataset((results, parallelism), out_file)
    return costs, np.sum(durations) / parallelism



