##
##   Code from Bansal and Weld paper aiweb.cs.washington.edu/ai/unkunk18
##
from __future__ import division
import os
import sys
import random
import multiprocessing as mp
import shutil
import numpy as np

#import config

small_fixed_prior = 0.5


def get_prior(prior_type, utility_model, k=None, clusters=None):
    prior = [0 for i in range(utility_model.num_inputs)]
    if prior_type == "optimal_cluster":
        assert k > 1
        cluster_prior = [0 for i in range(k)]
        cluster_unkunk_count = [0 for i in range(k)]
        cluster_cand_count = [0 for i in range(k)]
        for i in utility_model.get_cand_unkunks():
            cluster_cand_count[clusters[i]] += 1
            if utility_model._is_unkunk(i):
                cluster_unkunk_count[clusters[i]] += 1
        for i in range(k):
            if cluster_cand_count[i] != 0:
                cluster_prior[i] = cluster_unkunk_count[
                    i] / cluster_cand_count[i]

    for i in utility_model.get_cand_unkunks():
        if prior_type == "uniform":
            prior[i] = small_fixed_prior
        elif prior_type == "conf":
            prior[i] = 1 - utility_model.conf[i]
        elif prior_type == "optimal":
            prior[i] = int(utility_model._is_unkunk(i))
        elif prior_type == "optimal_cluster":
            prior[i] = cluster_prior[clusters[i]]
        else:
            print("Invalid prior type")
            raise Exception
#    if prior_type != "optimal":
#         print("diff from optimal", np.linalg.norm(
#             np.array(prior) - get_prior("optimal", utility_model)))
    return np.array(prior)


def update_foo(update_type, prior, solution, utilities, utility_model, k,
               clusters):
    new_prior = None
    if update_type == "noupdate":
        new_prior = prior
    elif update_type == "cluster":
        cluster_unkunk_count = [0 for i in range(k)]
        cluster_count = [0 for i in range(k)]
        
        ##
        ## Mistake?
        ##
            
        #for i in range(len(solution)):
        for i in solution:
            i_cluster = clusters[i]
            if utility_model._is_unkunk(i):
                cluster_unkunk_count[i_cluster] += 1
            cluster_count[i_cluster] += 1
        scale = 2
        new_prior = [(prior[i] * scale + cluster_unkunk_count[clusters[i]]) /
                     (1 * scale + cluster_count[clusters[i]])
                     for i in range(prior.shape[0])]
    elif update_type == "meta":
        pass
    else:
        print("invalid update type")
        raise Exception
    return np.array(new_prior)


def adap_greedy(prior_type,
                utility_model,
                budget,
                update_type,
                k=None,
                clusters=None):
    solution = []
    utilities = []
    inputs = set(range(utility_model.num_inputs))
    prior = get_prior(prior_type, utility_model, k=k, clusters=clusters)
    for b in range(min(budget, len(utility_model.get_cand_unkunks()))):
  
        sys.stdout.flush()
        new_prior = update_foo(
            update_type,
            prior,
            solution,
            utilities,
            utility_model,
            k=k,
            clusters=clusters)
        input_list = list(inputs)
        max_util, max_util_input_idx = utility_model.get_argmax_utility(
            solution, input_list, new_prior)
        max_util_input = input_list[max_util_input_idx]
        inputs.remove(max_util_input)
        solution.append(max_util_input)
        utilities.append(utility_model.get_utility(solution))
    assert np.all(np.diff(utilities) >= 0)  # ensure monotonicity
  #  print solution[:10]
  #  print utility_model.conf[solution[:10]]
    return solution, utilities


def most_uncertain(utility_model, budget):
    candidates = utility_model.get_cand_unkunks()
    candidates_sorted = sorted(candidates, key=lambda x: utility_model.conf[x])
    solution = candidates_sorted[:budget]
    utilities = []
    for b in range(budget):
        utility = utility_model.get_utility(solution[:(b + 1)])
        utilities.append(utility)
 ##   print solution[:10]
##    print utility_model.conf[solution[:10]]
    return solution, utilities


def mp_args_helper(args_tuple):
    foo, seed, args, kwargs = args_tuple
    kwargs["seed"] = seed
    return foo(*args, **kwargs)


# def multiple_runs(foo, num_runs, start_seed, outdir, num_processes, *args,
#                   **kwargs):
#     solution_list = []
#     utilities_list = []
#     if num_processes == 1:
#         for i in range(num_runs):
#             kwargs["seed"] = start_seed + i
#             _, utilities = foo(*args, **kwargs)
#             print i, utilities[-1],
#             utilities = np.array(utilities)
#             utilities_list.append(utilities)
#     else:
#         pool = mp.Pool(num_processes)
#         for i, (solution, utilities) in enumerate(
#                 pool.imap_unordered(mp_args_helper,
#                                     ((foo, start_seed + i, args, kwargs)
#                                      for i in range(num_runs)))):
#             utilities = np.array(utilities)
#             print i, utilities[-1],
#             utilities_list.append(utilities)
#             solution_list.append(np.array(solution))
#             sys.stdout.flush()
#     print
#     final_array = np.array(utilities_list)
#     outfile = os.path.join(outdir, "%d_%d.npy" % (start_seed,
#                                                   start_seed + num_runs))
#     print "saving", outfile
#     np.save(outfile, final_array)

#     # final_solution = np.array(solution_list)
#     # outfile = os.path.join("kaggle_src", "uubsol.npy")
#     # np.save(outfile, solution_list[0])
#     save_query_src(utilities_list[0], solution_list[0],
#                    "%s_uus_%s" % (config.args.dataset, args[0]))
    


def save_query_src(utilities, solution, outdir, dataset):
    prev_util = 0
    uu_list = []
    for i in range(50):
        if utilities[i] > prev_util:
            uu_list.append(solution[i])
            prev_util = utilities[i]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if dataset == "kaggle13":
        for i, uu in enumerate(uu_list):
            shutil.copyfile("%s_src/%d.jpeg" % (dataset, uu),
                            os.path.join(outdir, "%d.jpeg" % i))
    else:
        srcfile = open("%s_src.txt" % dataset, "rb")
        docs = [line for line in srcfile]
        srcfile.close()
        ofile = open("%s.txt" % outdir, "wb")
        for i, uu in enumerate(uu_list):
            ofile.write(docs[uu] + "\n")


def bandit_solution(algo,
                    utility_model,
                    k,
                    clusters,
                    budget,
                    epsilon=0.1,
                    discount=0.8,
                    prior=0.1,
                    utility = 'submod',
                    seed=None):
    random.seed(seed)
    solution = []
    utilities = []
    arms = [[] for i in range(k)]
    arm_rewards = [[] for i in range(k)]
    arm_times = [[] for i in range(k)]
    for i in utility_model.get_cand_unkunks():
        arms[clusters[i]].append(i)
    arm_initial_size = [len(arms[i]) for i in range(k)]

    def disc(arm, j, t):
        if algo == "discucb":
            return np.power(discount, t - j)
        elif algo == "uub":        
            
            return (arm_initial_size[arm] - len(arm_times[arm])) / (
                arm_initial_size[arm] - len(
                    [i for i in arm_times[arm] if i < j]))
        else:
            raise Exception

    def n(arm):
        return len(arm_times[arm])

    def u(arm):
        return np.sum(arm_rewards[arm]) / n(arm)

    def u_exp_true(arm):
        if len(arms[arm]) == 0:
            return 0
        return utility_model.get_uu_prob(arms[arm])

    def u_exp(arm):
        if len(arms[arm]) == 0:
            return 0
        mean_reward = utility_model.get_mean_max_possible_utility(
            solution, arms[arm])
        return mean_reward

    def n_disc(arm, t):
        return np.sum([disc(arm, j, t) for j in arm_times[arm]])

    def u_disc(arm, t):
        
     #   print("u_disc Arm rewards (arm: ", str(arm), ")")
      #  print([arm_rewards[arm][i] for i in range(len(arm_times[arm]))])
        
        toReturn = np.sum([
            disc(arm, arm_times[arm][i], t) * arm_rewards[arm][i]
            for i in range(len(arm_times[arm]))
        ])
        
     #   print("u_disc arm: ", str(arm), " t: ", str(t+1), " toReturn: ")
     #   print(toReturn)
        
        
        return np.sum([
            disc(arm, arm_times[arm][i], t) * arm_rewards[arm][i]
            for i in range(len(arm_times[arm]))
        ]) / n_disc(arm, t)

    def isvalid(arm):
        return int(len(arms[arm]) > 0)

    ne_arms = [i for i in range(k) if isvalid(i) is 1]

    for t in range(budget):
        if t < len(ne_arms) and algo != "optimal" and algo != "submodgreedy":
            curr_arm = ne_arms[t]
        else:
            if algo == "ucb":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: (u(i) + np.sqrt(2 * np.log(t + 1) / n(i))))
            elif algo == "discucb" or algo == "uub":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: (u_disc(i, t) + np.sqrt(2 * np.log(t + 1) / n_disc(i, t))))
         #       print("PartOne: " + str(t + 1))
        #        print([u_disc(i,t) for i in range(k) if isvalid(i) is 1])
                
      #          print("PartTwo: " + str(t + 1))
     #           print([np.sqrt(2 * np.log(t + 1) / n_disc(i, t)) for i in range(k) if isvalid(i) is 1])
       #         print("n_disc: " + str(t+1))
        #        print([n_disc(i, t) for i in range(k) if isvalid(i) is 1])
                
                
                
                
          #      print("Time :" + str(t + 1))
           #     print([(u_disc(i, t) + np.sqrt(2 * np.log(t + 1) / n_disc(i, t))) for i in range(k) if isvalid(i) is 1])
                
            elif algo == "epsilongreedy":
                if random.uniform(0, 1) > epsilon:
                    curr_arm = max(
                        [i for i in range(k) if isvalid(i) is 1],
                        key=lambda i: u(i))
                else:
                    curr_arm = random.choice(
                        [i for i in range(k) if isvalid(i) is 1])
            elif algo == "random":
                curr_arm = random.choice(
                    [i for i in range(k) if isvalid(i) is 1])
            elif algo == "optimal":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: u_exp_true(i))
            elif algo == "submodgreedy":
                probs = [((np.nonzero(arm_rewards[i])[0].shape[0] + 1) /
                          (n(i) + 100 * prior)) for i in range(k)]
                scores = [probs[i] * u_exp(i) for i in range(k)]
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: scores[i])
            else:
              #  print algo, "not implemented"
                raise Exception
        rand_idx = 0
        if len(arms[curr_arm]) != 1:
            if utility == "fixed":
                rand_idx = random.randint(0, len(arms[curr_arm]) - 1)
            elif utility == "submod":
                _, rand_idx = utility_model.get_argmax_utility(
                    solution, arms[curr_arm], prior=None)
        rand_sample = arms[curr_arm][rand_idx]
        arms[curr_arm].remove(rand_sample)

        solution.append(rand_sample)
        utilities.append(utility_model.get_utility(solution))

        # u and b
        reward = utilities[-1]
        if len(utilities) > 1:
            reward = utilities[-1] - utilities[-2]
        reward = int(reward > 1e-2)
        arm_rewards[curr_arm].append(reward)
        
        
        
 #       print("curr_arm: ", str(curr_arm), " arm rewards :" + str(t))
  #      print(arm_rewards[curr_arm])
        arm_times[curr_arm].append(t)
      #  print("arm times :" + str(t))
      #  print(arm_times[curr_arm])

    return solution, utilities

def bandit_solution_SDR(algo,
                    utility_model,
                    k,
                    clusters,
                    budget,
                    epsilon=0.1,
                    discount=0.8,
                    prior=0.1,
                    utility = 'submod',
                    seed=None):
    random.seed(seed)
    solution = []
    utilities = []
    arms = [[] for i in range(k)]
    arm_rewards = [[] for i in range(k)]
    arm_conf = [[] for i in range(k)]
    arm_corr = [[] for i in range(k)]
    arm_times = [[] for i in range(k)]
    for i in utility_model.get_cand_unkunks():
        arms[clusters[i]].append(i)
    arm_initial_size = [len(arms[i]) for i in range(k)]

    def disc(arm, j, t):
        if algo == "discucb":
            return np.power(discount, t - j)
        elif algo == "uub":        
            
            return (arm_initial_size[arm] - len(arm_times[arm])) / (
                arm_initial_size[arm] - len(
                    [i for i in arm_times[arm] if i < j]))
        else:
            raise Exception

    def n(arm):
        return len(arm_times[arm])

    def u(arm):
        return np.sum(arm_rewards[arm]) / n(arm)

    def u_exp_true(arm):
        if len(arms[arm]) == 0:
            return 0
        return utility_model.get_uu_prob(arms[arm])

    def u_exp(arm):
        if len(arms[arm]) == 0:
            return 0
        mean_reward = utility_model.get_mean_max_possible_utility(
            solution, arms[arm])
        return mean_reward

    def n_disc(arm, t):
        return np.sum([disc(arm, j, t) for j in arm_times[arm]])

    def u_disc(arm, t):
        
     #   print("u_disc Arm rewards (arm: ", str(arm), ")")
      #  print([arm_rewards[arm][i] for i in range(len(arm_times[arm]))])
        
        toReturn = np.sum([
            disc(arm, arm_times[arm][i], t) * arm_rewards[arm][i]
            for i in range(len(arm_times[arm]))
        ])
        
     #   print("u_disc arm: ", str(arm), " t: ", str(t+1), " toReturn: ")
     #   print(toReturn)
        
        
        return np.sum([
            disc(arm, arm_times[arm][i], t) * arm_rewards[arm][i]
            for i in range(len(arm_times[arm]))
        ]) / n_disc(arm, t)

    def isvalid(arm):
        return int(len(arms[arm]) > 0)

    ne_arms = [i for i in range(k) if isvalid(i) is 1]

    for t in range(budget):
        if t < len(ne_arms) and algo != "optimal" and algo != "submodgreedy":
            curr_arm = ne_arms[t]
        else:
            if algo == "ucb":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: (u(i) + np.sqrt(2 * np.log(t + 1) / n(i))))
            elif algo == "discucb" or algo == "uub":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: (u_disc(i, t) + np.sqrt(2 * np.log(t + 1) / n_disc(i, t))))
         #       print("PartOne: " + str(t + 1))
        #        print([u_disc(i,t) for i in range(k) if isvalid(i) is 1])
                
      #          print("PartTwo: " + str(t + 1))
     #           print([np.sqrt(2 * np.log(t + 1) / n_disc(i, t)) for i in range(k) if isvalid(i) is 1])
       #         print("n_disc: " + str(t+1))
        #        print([n_disc(i, t) for i in range(k) if isvalid(i) is 1])
                
                
                
                
          #      print("Time :" + str(t + 1))
           #     print([(u_disc(i, t) + np.sqrt(2 * np.log(t + 1) / n_disc(i, t))) for i in range(k) if isvalid(i) is 1])
                
            elif algo == "epsilongreedy":
                if random.uniform(0, 1) > epsilon:
                    curr_arm = max(
                        [i for i in range(k) if isvalid(i) is 1],
                        key=lambda i: u(i))
                else:
                    curr_arm = random.choice(
                        [i for i in range(k) if isvalid(i) is 1])
            elif algo == "random":
                curr_arm = random.choice(
                    [i for i in range(k) if isvalid(i) is 1])
            elif algo == "optimal":
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: u_exp_true(i))
            elif algo == "submodgreedy":
                probs = [((np.nonzero(arm_rewards[i])[0].shape[0] + 1) /
                          (n(i) + 100 * prior)) for i in range(k)]
                scores = [probs[i] * u_exp(i) for i in range(k)]
                curr_arm = max(
                    [i for i in range(k) if isvalid(i) is 1],
                    key=lambda i: scores[i])
            else:
              #  print algo, "not implemented"
                raise Exception
        rand_idx = 0
        if len(arms[curr_arm]) != 1:
            if utility == "fixed":
                rand_idx = random.randint(0, len(arms[curr_arm]) - 1)
            elif utility == "submod":
                _, rand_idx = utility_model.get_argmax_utility(
                    solution, arms[curr_arm], prior=None)
        rand_sample = arms[curr_arm][rand_idx]
        arms[curr_arm].remove(rand_sample)

        solution.append(rand_sample)
        utilities.append(utility_model.get_utility(solution))

        # u and b
        reward = utilities[-1]
        if len(utilities) > 1:
            reward = utilities[-1] - utilities[-2]
        reward = int(reward > 1e-2)
        arm_rewards[curr_arm].append(reward)
        arm_conf[curr_arm].append()
        
        
        
 #       print("curr_arm: ", str(curr_arm), " arm rewards :" + str(t))
  #      print(arm_rewards[curr_arm])
        arm_times[curr_arm].append(t)
      #  print("arm times :" + str(t))
      #  print(arm_times[curr_arm])

    return solution, utilities