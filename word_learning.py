import numpy as np
import matplotlib.pyplot as plt
import collections
import csv
import cPickle as pickle
import os
import sys
import operator    # used to get the key having maximum value in a dictionary

"""
Sample small hypothesis space created to  
make it easy to analyze the model


        [o]           (height = 1)
         |
        [a]           (height = 0.8)
      /    \
    /        \
  [b]       [e]       (height = 0.3)
  / \       / \
[c] [d]   [f] [g]     (height = 0)
 1  4      5   7
 2         6   8
 3             9
 
 
 
"""
################################################# Global Variables ###########################################  


def read_csv(filename):
    """
    Reads a csv file to construct the hypothesis
    space and the node map. Specifically, it constructs
    one list and three dictionaries explained below: 
    nodes (list) : list of nodes in the hypothesis space
    heights (dict) : stores the height of each node
    parents (dict) : stores the immediate parent of each node
    node_map (dict) : maps each node to numbers from [0, num of nodes-1 ]
    
    Example output for small hypothesis space given above
    nodes = ['a','b','c','d','e','f','g']
    heights = {'o':1, 'a':0.8, 'b':0.3, 'e':0.3, 'c':0, 'd':0, 'f':0, 'g':0}
    parents = {'o':FALSE, 'a':'o', 'b':'a', 'e':'a', 'c':'b', 'd':'b', 'f':'e', 'g':'e',
                  1:'c', 2:'c', 3:'c', 4:'d', 5:'f', 6:'f', 7:'g', 8:'g', 9:'g'}
    node_map = {0: 'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g'}
    """
    
    global nodes
    global heights
    global parents
    global node_map
    nodes = []
    heights = {}
    parents = {}
    node_map = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if i==0:
                i = i+1
                continue
            # if the node name is an integer, then it is an example/data
            elif row[0].isdigit():
                parents[int(row[0])] = row[2]
            else:
                heights[row[0]] = float(row[1])
                parents[row[0]] = row[2]
                if i!=1:
                    node_map[int(row[3])] = row[0]
                    nodes.append(row[0])
                
                i = i+1

################################################## Node Methods ############################################################


def is_child(node1, node2):
    """
    Checks whether node1 is a descendent of node2
    Parameters
    ----------
    node: string
    Name of a node in the hypothesis space 'a', 'LL' etc.
    
    Returns
    -------
    A boolean value 
    """
    
    if (node1 == 'FALSE'):
        return False
    elif (node1 == node2):
        return True
    else:
        return is_child(node_parent(node1), node2)
        
def node_parent(node):
    """
    Returns the parent of node
    Parameters
    ----------
    node: string
    Name of a node in the hypothesis space 'a', 'LL' etc.
    
    Returns
    -------
    node: string
    Name of a node in the hypothesis space 'a', 'LL' etc.
    """
    
    return parents[node]


def node_height(node):
    """
    Returns the height of node
    Parameters
    ----------
    node: string
    Name of a node in the hypothesis space 'a', 'LL' etc.
    """
    
    return heights[node]
    

# function to calculate the prior of all the nodes   
# mainly need a list for multinomial distribution
def cal_prior(nodes):
    """ 
    Calculates the prior for all nodes
    Parameters
    ----------
    nodes: A list of nodes 
    
    Returns
    -------
    weights: A list containing the prior weights for all nodes
    """

    # define prior probabilities
    raw_weights = [(node_height(node_parent(x)) - node_height(x)) for x in nodes]
    #raw_weights = [1 for x in nodes]

    # normalize prior probabilities
    weights = [float(x) / sum(raw_weights) for x in raw_weights ]
    return weights


def raw_prior(node):
    """
    Returns the un-normalized prior value for node
    Parameters
    ----------
    node: string
        Name of a node in the hypothesis space 'a', 'LL' etc.
    """
    
    b = 1
    if node == 'J' or node == 'R' or node == 'T':
        b = beta   # basic-level bias
    return b*( node_height(node_parent(node)) - node_height(node) )
    

def prior(node):
    """
    Returns the normalized prior value for node
    Parameters
    ----------
    node: string
        Name of a node in the hypothesis space 'a', 'LL' etc.
    """
    
    # uncomment the following line for a uniform prior
    #return 1.0/len(nodes)   #uniform prior
    
    # sum of priors of all nodes
    prior_denom = sum([raw_prior(node_x) for node_x in nodes])
    # return the prior of the given node
    return raw_prior(node)/prior_denom


def raw_likelihood(node, data):
    """
    Returns the un-normalized likelihood value for node, given data
    Parameters
    ----------
    node: string
        Name of a node in the hypothesis space 'a', 'LL' etc.
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
    """
    
    n = len(data)
    return ((node_height(node) + epsilon)**-n)


def likelihood(node, data):
    """
    Returns the normalized likelihood value for node, given data
    Parameters
    ----------
    node: string
        Name of a node in the hypothesis space 'a', 'LL' etc.
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
    """
    
    # sum of likelihoods of all nodes
    likelihood_denom = sum([raw_likelihood(node_x, data) for node_x in nodes])
    # return the normalized likelihood of the given node
    return raw_likelihood(node, data)/likelihood_denom

 
def barchart(samples):
    """
    Creates a histogram from the samples   
    representing the posterior distribution
    """
    
    if sum(samples) == 0:
        print "No samples collected"
        return plt
    global final_prob    
    norm = [float(i)/sum(samples) for i in samples]
    final_prob = norm
    ind = np.arange(len(samples))
    width = 1                       
    plt.figure(figsize=(len(nodes), 6), facecolor='white')
    plt.bar(ind, norm, width, color='grey') #, yerr=menStd)
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Nodes', fontsize=16)
    plt.xticks(ind + width/2., nodes, fontsize=16)
    return plt


############################################## Coin toss ########################################################


def flip_coin(bias=0.5):
    """
    Returns the outcome of a biased coin
    Parameters
    ----------
    bias: float
        Probability of landing heads (i.e., an outcome of 1)
    """
    
    p1 = bias
    p0 = 1-p1
    return np.random.choice([0,1],p=[p0,p1])   


def coin_samples(n=1, function=None, bias=0.5):
    """
    Returns samples for multiple coin flips
    Parameters
    ----------
    n: integer
        Number of samples required
    function: python function name
        function to call to generate coinflips
    bias: float
        Probability of coin landing heads (i.e., an outcome of 1)
    """
    
    sample_points = []
    if function is None:
        return
    else:
        for x in range(n):
            if function==flip_coin:
                sample_points.extend([function(bias)])
            else:        
                sample_points.extend([function()])
    return sample_points


def get_coin_samples(num_samples=10000, bias=0.5):
    """
    Returns samples for multiple coin flips by calling coin_samples()
    Parameters
    ----------
    num_samples: integer
        Number of samples required
    bias: float
        Probability of coin landing heads (i.e., an outcome of 1)
    """
    samples = coin_samples(num_samples, function=flip_coin, bias=bias)
    return samples


def plot_coin_samples(samples):
    """
    Plots a histogram of coin samples by calling hist().
    Used to visualize the bias in the coin.
    """
    
    plt = hist(samples)
    plt.show()
    

def hist(samples):
    """
    Plots a histogram of coin samples
    """
    
    plt.figure()
    #plt.hist(samples,2,normed=1,facecolor='green', alpha=0.75)
    results, edges = np.histogram(samples, normed=True)
    binWidth = edges[1] - edges[0]
    plt.bar(edges[:-1], results*binWidth, binWidth)
    return plt


################################################## Rejection Sampling #############################################

## Global variables for rejection sampling    
total_samples = 0
rejected_samples = 0
accepted_samples = 0
samples_considred = 0


def rejection_sampling(acc_samples, prior_weights, data):
    """
    Returns accepted samples obtained by rejection sampling
    Parameters
    ----------
    acc_samples: int
        Number of accepted samples required
    prior_weights: list
        list of priors of all nodes in hypothesis space
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
    """
    
    result = data_samples(acc_samples, prior_weights, data)
    
    print "Rejection Sampling"
    print "Total Samples: ", total_samples
    print "Rejected Samples: ", rejected_samples
    print "Accepted Samples: ", total_samples - rejected_samples
    print "Samples Considred: ", samples_considred
    
    return result
    

def data_samples(acc_samples, prior_weights, data):
    """
    Returns accepted samples obtained by rejection sampling
    Parameters
    ----------
    acc_samples: int
        Number of accepted samples required
    prior_weights: list
        list of priors of all nodes in hypothesis space
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
    """
    
    result = init_result()
    # call draw_samples n (acc_samples) times
    for x in range(acc_samples):
        draw_Samples(prior_weights, data, result)
    return result
        

def draw_Samples(prior_weights, data, result):
    """
    Returns one accepted sample
    Parameters
    ----------
    prior_weights: list
        List of priors of all nodes in hypothesis space
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
    result: dict
        An empty dictionary for storing the resulting 
        samples. 
    """
    
    global total_samples
    global rejected_samples
    global samples_considred  # accepted samples added to the result
    
    #draw hypothesis according to the prior
    ind = np.where(np.random.multinomial(1, prior_weights))[0][0]
    node = nodes[ind]
    total_samples += 1
    
    if all( [is_child(x, node)for x in data] ):
        if flip_coin(likelihood(node, data)):
            samples_considred += 1
            result[node] += 1  
    else: 
        rejected_samples += 1
        draw_Samples(prior_weights, data, result) 
      
    
def init_result(): 
    """
    Returns an empty dictionary for storing the resulting 
    samples generated from rejection sampling or mcmc.
    
    The dictionary has keys as nodes and 
    values as the number of times the respective  
    nodes were predicted.
    """
    
    # dictionary containing final result
    result = collections.OrderedDict()
    for node in nodes:
        result[node] = 0
    return result    
        
    
def plot_result(result, title): 
    """
    Creates a histogram from the samples   
    representing the posterior distribution
    It calls barchart() to generate the histogram
    """
    
    samp = []
    for key,value in result.items():
        samp.extend([value])
    plt = barchart(samp)
    plt.title(title)
    #plt.show()
    plt.savefig('./mcmc_plots/' + title, fontsize=20)
    plt.close()
    
    
    
def get_prediction(result):
    """
    Result is a dictionary with keys as nodes and 
    values as the number of times the respective nodes 
    were predicted.
    
    Returns the node name e.g., 'b' which is the MAP estimate
    """
    
    return max(result.iteritems(), key=operator.itemgetter(1))[0]
    

########################################### Markov Chain Monte Carlo ###########################################



def target(node, data):
    """
    Defines the target distribution we want to 
    sample from i.e., the posterior distribution.
    """
    
    # find the mapping to letters
    node = node_map[node]
    pri = prior(node)
    if all( [is_child(x, node)for x in data] ):
        lik = likelihood(node, data)
    else:
        lik = 0
    return pri*lik
    
    
# symmetric proposal function
# equally likely to propose one number higher or lower
def symm_pfun(x):
    """
    Defines the proposal function
    1) Equally likely to propose one number higher or lower
    2) A variant of symmetric random proposal
    """
    
    return np.random.choice(range(len(nodes)))
    # uncomment the following and comment out the  
    # line above for equally likely proposal
    """
    if flip_coin(0.5):
        return x-1
    else:
        return x+1 
    """       
        

def symm_pdist(x):   
    """
    Defines a symmetric proposal distribution
    """
    
    return 0.5


def normal_pfun(sigma, mu):
    """
    Defines a proposal function which is a variant of
    the normal distribution. 
    
    Returns the sample from normal distributions parsed as an integer
    """
    
    # return a sample from the normal distribution
    return int(sigma * np.random.randn() + mu)
       
       
# probability of x2 given x1    
def normal_pdist(x, mu, sigma):
    """
    Defines the probability density for the normal distribution
    Returns the density value for a given x.
    """ 
    return ( 1/np.sqrt(2*np.pi*sigma**2) ) *  np.exp(-(x-mu)**2 / 2*sigma**2)
    
    
   
def mcmc_symm(num_samples, data):
    """
    Returns num_samples samples obtained by mcmc
    Parameters
    ----------
    num_samples: int
        Number of samples required
    data: list
        List of integers representing observed examples e.g., [1, 2, 3]
   
   Returns
    -------
     result: dict
        An dictionary storing the resulting samples 
        corresponding to nodes. 
    """
    
    z = np.zeros(num_samples)
    a = np.zeros(num_samples)
    z[0] = 0
    sd = 1
    for i in range(2, num_samples):
        
        x = z[i-1]   # old state
        y = symm_pfun(x)   # propose a new state
        #y = normal_pfun(sigma=sd, mu=x)
        
        # accept new y with prob
        if y >= 0 and y < len(nodes):
            rtarget = target(y, data)/target(x, data)         # target ratio
            #rproposal = normal_pdist(y, x, sigma=sd) / normal_pdist(x, y, sigma=sd)   # proposal ratio
            rproposal = symm_pdist(x) / symm_pdist(y)
            p = rtarget*rproposal
        else:
            p = 0    
            
        # generate a u from the uniform distribution     
        u = np.random.uniform(0,1)
        if u < min(p, 1):
            # accept the proposal
            z[i] = y
            a[i] = 1
        else:
            z[i] = x
            a[i] = 0
    
    
    #np.save("./result_files/normal_small4.npy", z)
    #np.save("./result_files/normal_small_acc4.npy", a)
    
    # removing the first 10,000 samples
    z = z[10000:]
    # introduce a lag of 50
    z = z[np.arange(0, num_samples-10000, 50)]
    
    val = {'a':a, 'z':z}
    
    result  = mcmc_result(z)
    return(result)
    
    
    
def credible_interval(z):
    """
    Computes the 95% credible interval using percentiles
    z: list
        List of samples drawn from mcmc.
    """
    print "C Interval: ", np.percentile(z, 2.5)
    print "C Interval: ", np.percentile(z, 97.5)
 

    
def mcmc_result(state_samples):
    """
    Creates the result dictionary using the samples from mcmc
    Parameters
    ----------
    state_samples: list 
        List of samples drawn through mcmc
        
    Returns
    -------
     result: dict
        An dictionary storing the resulting samples 
        corresponding to nodes    
    """
    
    result = init_result()
    for sample in state_samples:
        node = node_map[sample]
        result[node] += 1
    return result    
    
    


########################################## Probability of Generalization Plots #########################################

def p_parents(x):
    """
    Finds the total probability of x and 
    all its' parents recursively
    
    Parameters
    ----------
    x: int
        The index of a node
    """
    
    node = node_map[x]
    node_list = []
    pg_sup = 0
    if node!='root':
        node_list.append(node)
        node = node_parent(node)
        
    for node in node_list: 
        for key, value in node_map.iteritems():   
            if value == node:
                pg_sup += final_prob[key]
                break
    return pg_sup
    
    
def three_sub(sub, basic, sup, title="Three sub"):
    """
    Computes the probability of generalization when 
    three examples from the subordinate category are observed
    
    Parameters
    ----------
    sub: int
        The index of the subordinate category node
    basic: int
        The index of the basic category node
    sup: int 
        The index of the superordinate category node
    title: string
        The title fo the plot (also used as filename to save the plot)
    
    Returns
    -------
    pg_list: list
        A list containing the probabilities of generalization for an 
        example 'y' from the subordinate, basic and superordiante categories
    """
    
    pg_sub = np.sum(final_prob)
    pg_basic = p_parents(basic)
    pg_sup = p_parents(sup)
    pg_list = [pg_sub, pg_basic, pg_sup]
    pg_barplot(pg_list, title)
    return pg_list

    
def three_basic(sub, basic, sup, title="Three basic"):
    """
    Computes the probability of generalization when 
    three examples from the basic category are observed
    
    Parameters
    ----------
    sub: int
        The index of the subordinate category node
    basic: int
        The index of the basic category node
    sup: int 
        The index of the superordinate category node
    title: string
        The title fo the plot (also used as filename to save the plot)
    
    Returns
    -------
    pg_list: list
        A list containing the probabilities of generalization for an 
        example 'y' from the subordinate, basic and superordiante categories
    """
    
    pg_sub = np.sum(final_prob)
    pg_basic = np.sum(final_prob)
    pg_sup = p_parents(sup)
    pg_list = [pg_sub, pg_basic, pg_sup]
    pg_barplot(pg_list, title)
    return pg_list
    
  
def three_sup(sub, basic, sup, title="Three sup"):
    """
    Computes the probability of generalization when 
    three examples from the subordinate category are observed
    
    Parameters
    ----------
    sub: int
        The index of the subordinate category node
    basic: int
        The index of the basic category node
    sup: int 
        The index of the superordinate category node
    title: string
        The title fo the plot (also used as filename to save the plot)
    
    Returns
    -------
    pg_list: list
        A list containing the probabilities of generalization for an 
        example 'y' from the subordinate, basic and superordiante categories
    """
    
    pg_sub = np.sum(final_prob)
    pg_basic = np.sum(final_prob)
    pg_sup = np.sum(final_prob)
    pg_list = [pg_sub, pg_basic, pg_sup]
    pg_barplot(pg_list, title)
    return pg_list


def pg_barplot(pg_list, title):
    """
    Creates a histogram to visualize the probability
    of generalization and saves it in a directory
    
    Parameters
    ----------
    pg_list: list
        A list containing the probabilities of generalization for an 
        example 'y' from the subordinate, basic and superordiante categories
    title: string
        The title fo the plot (also used as filename to save the plot)
    """
    
    ind = np.arange(len(pg_list))
    width = 1                       
    plt.figure(figsize=(5, 6), facecolor='white')
    plt.bar(ind, pg_list, width, color='grey') #, yerr=menStd)
    plt.xlabel('Categories', fontsize=11)
    plt.ylabel('Probability of generalization', fontsize=12)
    plt.title(title)
    plt.xticks(ind + width/2., ['sub', 'basic', 'super'])
    #plt.show()
    plt.savefig('./generalization_plots/' + title, fontsize=12)
    plt.close()
    return plt


###################################################### Data Processing Methods ##########################################################

def vegetables(flag):
    """
    Returns the data (examples) from the vegetable cluster in the 
    large hypothesis space. The category and number of examples 
    depends on the flag argument
    """
     
    if flag == '1sub':
        #1 subordinate => B (33)
        data = [16]
    
    if flag == '3sub':
        # 3 subordinate => B (33)
        data = [16, 17, 18]    # observed data
    
    if flag == '3basic':
        # 3 basic => J (29)
        data = [16, 21, 22]    # observed data
    
    if flag == '3sup':
        # 3 superordinate => BB (27)
        data = [16, 25, 26]    # observed data
    
    return data
    

def vehicles(flag):
    """
    Returns the data (examples) from the vehicle cluster in the 
    large hypothesis space. The category and number of examples 
    depends on the flag argument
    """
    
    if flag == '1sub':
        #1 subordinate => E (22)
        data = [31]
    
    if flag == '3sub':
        # 3 subordinate => E (22)
        data = [31, 32, 33]    # observed data
    
    if flag == '3basic':
        # 3 basic => T (17)
        data = [31, 36, 37]    # observed data
    
    if flag == '3sup':
        # 3 superordinate => HH (14)
        data = [31, 40, 41]    # observed data
    
    return data
    

def animals(flag):
    """
    Returns the data (examples) from the animal cluster in the 
    large hypothesis space. The category and number of examples 
    depends on the flag argument
    """
    
    if flag == '1sub':
        #1 subordinate => A (11)
        data = [1]
    
    if flag == '3sub':
        # 3 subordinate => A (11)
        data = [1, 2, 3]    # observed data
    
    if flag == '3basic':
        # 3 basic => R (7)
        data = [1, 6, 7]    # observed data
    
    if flag == '3sup':
        # 3 superordinate => JJ (2)
        data = [1, 10, 11]    # observed data
    
    return data
    

            
def automate_result():
    """
    Automates creation of generalization plots for the 
    large hypothesis space. Saves all the plots in a directory.
    """
    
    # number of samples for mcmc
    num_samples = 50000 
    
    # vegetables
    data = vegetables('1sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vegetable: 1 sub")
    vegetable_1sub = three_sub(33, 29, 27, 'Vegetable: 1 sub')
    
    data = vegetables('3sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vegetable: 3 sub")
    vegetable_3sub = three_sub(33, 29, 27, "Vegetable: 3 sub")
    
    data = vegetables('3basic')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vegetable: 3 basic")
    vegetable_3basic = three_basic(33, 29, 27, 'Vegetable: 3 basic')
    
    data = vegetables('3sup')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vegetable: 3 sup")
    vegetable_3sup = three_sup(33, 29, 27, 'Vegetable: 3 sup')
   
    
    # Vehicles
    data = vehicles('1sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vehicle: 1 sub")
    vehicle_1sub = three_sub(22, 17, 14, "Vehicle: 1 sub")
    
    data = vehicles('3sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vehicle: 3 sub")
    vehicle_3sub = three_sub(22, 17, 14, "Vehicle: 3 sub")
    
    data = vehicles('3basic')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vehicle: 3 basic")
    vehicle_3basic = three_basic(22, 17, 14, "Vehicle: 3 basic")
    
    data = vehicles('3sup')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Vehicle: 3 sup")
    vehicle_3sup = three_sup(22, 17, 14, "Vehicle: 3 sup")
    
    
    # Animals
    data = animals('1sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Animal: 1 sub")
    animal_1sub = three_sub(11, 7, 2, "Animal: 1 sub")
    
    data = animals('3sub')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Animal: 3 sub")
    animal_3sub = three_sub(11, 7, 2, "Animal: 3 sub")
    
    data = animals('3basic')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Animal: 3 basic")
    animal_3basic = three_basic(11, 7, 2, "Animal: 3 basic")
    
    data = animals('3sup')
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "Animal: 3 sup")
    animal_3sup = three_sup(11, 7, 2, "Animal: 3 sup")
    
    
    data = {
            'vegetable_1sub' : vegetable_1sub,
            'vegetable_3sub' : vegetable_3sub,
            'vegetable_3basic' : vegetable_3basic,
            'vegetable_3sup' : vegetable_3sup,
    
            'vehicle_1sub' : vehicle_1sub,
            'vehicle_3sub' : vehicle_3sub,
            'vehicle_3basic' : vehicle_3basic,
            'vehicle_3sup' : vehicle_3sup, 
    
            'animal_1sub' : animal_1sub,
            'animal_3sub' : animal_3sub,
            'animal_3basic' : animal_3basic,
            'animal_3sup' : animal_3sup, 
            }
            
    saveto_pickle(data)
 

def automate_result_small():
    """
    Automates creation of generalization plots for the 
    small hypothesis space. Saves all the plots in a directory.
    """
    
    # number of samples for mcmc   
    num_samples = 50000 
    
    # left branch
    data = [1]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "left branch: 1 sub")
    left_branch_1sub = three_sub(2, 1, 0, 'left branch: 1 sub')
    
    data = [1, 2, 3]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "left branch: 3 sub")
    left_branch_3sub = three_sub(2, 1, 0, "left branch: 3 sub")
    
    data = [1, 2, 4]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "left branch: 3 basic")
    left_branch_3basic = three_basic(2, 1, 0, 'left branch: 3 basic')
    
    data = [1, 4, 5]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "left branch: 3 sup")
    left_branch_3sup = three_sup(2, 1, 0, 'left branch: 3 sup')
   
    
    # right branch
    data = [7]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "right branch: 1 sub")
    right_branch_1sub = three_sub(6, 4, 0, "right branch: 1 sub")
    
    data = [7, 8, 9]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "right branch: 3 sub")
    right_branch_3sub = three_sub(6, 4, 0, "right branch: 3 sub")
    
    data = [7, 8, 5]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "right branch: 3 basic")
    right_branch_3basic = three_basic(6, 4, 0, "right branch: 3 basic")
    
    data = [7, 5, 4]
    result = mcmc_symm(num_samples=num_samples, data=data)
    plot_result(result, "right branch: 3 sup")
    right_branch_3sup = three_sup(6, 4, 0, "right branch: 3 sup")
    
    
    data = {
            'left_branch_1sub' : left_branch_1sub,
            'left_branch_3sub' : left_branch_3sub,
            'left_branch_3basic' : left_branch_3basic,
            'left_branch_3sup' : left_branch_3sup,
    
            'right_branch_1sub' : right_branch_1sub,
            'right_branch_3sub' : right_branch_3sub,
            'right_branch_3basic' : right_branch_3basic,
            'right_branch_3sup' : right_branch_3sup, 

            }
            
    saveto_pickle(data)
    
    
def saveto_pickle(data):
    """
    Writes the data to a pickle file and saves it.
    
    Parameters
    ----------
    data: dictionary
        A dictionary containing attributes and their values
    """
    
    # filename was passed as an argument to the script
    fname = sys.argv[1]  
    pickle.dump(data, open(fname, 'wb'))
    print ("pickle complete")
    print (fname)

        
#######################################################################################################################  
 
# global constant parameter epsilon  
epsilon = 0.00   # use 0.05 for small hypothesis space 

# parameter to fit to the experimental data
beta = 40   #only for mcmc and not for rejection sampling


def main():
    np.set_printoptions(threshold=np.nan)
    
    # Check whether the name of the pickle file  
    # was provided as an argument
    if len(sys.argv) > 1:
        pass 
    else:
        print "Require a file to store the output data, square_length and label_index"
        exit(0)
 
    
    # read data from the csv file to construct
    # nodes, heights, parents and node_maps lists/dictionaries
    read_csv('full_space.csv')
    automate_result()
    #automate_result_small() 
    
    
    ### small space ###
    # 1 subordinate => c (2)
    #data = [1]
    
    # 3 subordinate => c (2)
    #data = [1, 2, 3]    # observed data
    
    # 3 basic => b (1)
    #data = [1, 2, 4]    # observed data
    
    # 3 superordinate => a (0)
    #data = [1, 4, 5]    # observed data
    
    
    ### full space ###
    # 1 subordinate => F (32)
    #data = [22]
    
    # 3 subordinate => F (32)
    #data = [22, 16, 19]    # observed data
    
    # 3 basic => J (29)
    #data = [21, 24, 19]    # observed data
    
    # 3 superordinate => EE (26)
    #data = [21, 28, 27]    # observed data
    
    
    ### Rejection Sampling
    #prior_weights = cal_prior(nodes)
    #result = rejection_sampling(50000, prior_weights, data)
    #plot_result(result, "Rejection Sampling")
    
    ### MCMC
    #result = mcmc_symm(num_samples=50000, data=data)
    #prediction = get_prediction(result)
    #plot_result(result, "MCMC: full space, 1 sub, data=[22], Beta=40")
    #pg_list = three_sup(2, 1, 0, "Animal: 1 sub")

    ### Generating coin samples
    #samples = get_coin_samples(num_samples=10000, bias=0.8)
    #plot_coin_samples(samples=samples)
    
       
if __name__ == "__main__": main()
