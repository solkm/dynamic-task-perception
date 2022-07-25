#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:29:50 2021

@author: Sol
"""

from psychrnn.tasks.task import Task
import numpy as np
rng = np.random.default_rng(123)

"""
Within this file define task classes that can be called flexibly, including:
  * PerceptionModule: Takes in an explicit task input as well as DSL, DSF perceptual inputs.
  * StimHist_constant: A "K trials back" trial history model, with stimulus and reward history as constant inputs.
      - Can have data randomly drawn from one of two datasets at each trial.
"""
class PerceptionModule(Task):
    def __init__(self, dt, tau, T, N_batch, dat, in_noise=0.01, const_noise=0.0):
        super().__init__(3, 4, dt, tau, T, N_batch)
        self.dat = dat
        self.in_noise = in_noise
        self.const_noise = const_noise
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        dat = self.dat

        i = np.random.random_integers(0,len(dat['r_CHOICE'][0])-1)
            
        params = dict()
        params['dsl'] = dat['r_DSL'][0][i]
        params['dsf'] = dat['r_DSF'][0][i]
        params['task'] = dat['r_TASK'][0][i]
        params['correct'] = dat['r_CORRECT'][0][i]
        params['const_noise'] = self.const_noise*np.random.randn(2)
        
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        in_noise = self.in_noise
        onset = self.T / 4.0 #T is trial length in ms, onset is at T/4
        stim_dur = self.T / 2.0 #stim is on between T/4 and 3T/4
    
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        task = params['task']
        correct = params['correct']
        cn = params['const_noise']
        # ----------------------------------
        # Compute values
        # ----------------------------------
        x_t[2] += task
        if onset < time < onset + stim_dur:
            x_t[0] += dsl + cn[0]
            x_t[1] += dsf + cn[1]
    
        if time > onset + stim_dur + 20:          
            y_t[correct-1] = 1.
    
        if time < onset+ stim_dur: # non-response period
            mask_t = np.zeros(self.N_out)
        
        return x_t, y_t, mask_t
#
#
#   
class StimHist_constant(Task):
    """
    Parameters
    ----------
    dt (float): 
        The simulation timestep.
    tau (float): The intrinsic time constant of neural state decay.
    T (float): The length of a trial.
    N_batch (int): The number of trials in a batch.
    N_in (int): The input dimensionality. Set to 4*(K-1)+2: dsl, dsf, choice, reward for each past trial; dsl, dsf for the current trial.
    N_out (int): The output dimensionality. Set to 4.
    K (int): The total number of trials to consider, including the current trial and K-1 history trials.
    dat1 (dict): The source data.
    in_noise (float, optional): The input noise parameter, to scale Gaussian noise generated at each timestep. The default is 0.01.
    const_noise (float, optional): The visual stimulus noise parameter, to scale Gaussian noise added to visual inputs, generated at each trial. The default is 0.0.
    dat2 (dict, optional): A second dataset, where one of the two datasets is chosen at random at each trial.
    trainChoice (bool, optional): True if the network should be trained on the monkey's choice (rather than the correct choice). The default is False.
    nDist(bool, optional): True if stimulus inputs are to be drawn from the normalized decoded perceptual discriminability dataset. The default is False.
    gendat (bool, optional): True if trial parameters are to be drawn from a generated test dataset. The default is False.
    """
    def __init__(self, dt, tau, T, N_batch, K, dat1, in_noise=0.01, const_noise=0.0, dat2=None, trainChoice=False, nDist=False, gendat=False):
        N_in = 4*(K-1) + 2
        N_out = 4
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat1 = dat1
        self.dat2 = dat2
        self.in_noise = in_noise
        self.const_noise = const_noise
        self.trainChoice = trainChoice
        self.nDist = nDist
        self.gendat = gendat
        
    def generate_trial_params(self, batch, trial):
        """"
        Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.

        Returns:
            dict: Dictionary of trial parameters.

        """
        if self.dat2 is not None:
            p_dat2 = 0.5
            if np.random.rand() < p_dat2:
                dat = self.dat2
            else:
                dat = self.dat1
        else:
            dat = self.dat1
        
        K = self.K
        
        if self.gendat == False:
            
            i = np.random.randint(0, len(dat['r_CHOICE'][0])-(K+1))
        
            if self.nDist == True:
                while np.isnan(dat['r_NDSL'][0][i:i+K]).any()==True or np.isnan(dat['r_NDSF'][0][i:i+K]).any()==True:
                    i = np.random.random_integers(0, len(dat['r_CHOICE'][0])-(K+1))
            # ----------------------------------
            # Define parameters of a trial
            # ----------------------------------
    
            params = dict()
            params['choice'] = dat['r_CHOICE'][0][i:i+K]
            params['correct'] = dat['r_CORRECT'][0][i:i+K]
            params['task'] = dat['r_TASK'][0][i:i+K]
            params['const_noise'] = self.const_noise*np.random.randn(2*K)
            
            if self.nDist == False:
                params['dsl'] = dat['r_DSL'][0][i:i+K]
                params['dsf'] = dat['r_DSF'][0][i:i+K]
            else:
                params['ndsl'] = dat['r_NDSL'][0][i:i+K] #normalized distance from perceptual discrimination hyperplane
                params['ndsf'] = dat['r_NDSF'][0][i:i+K]
                params['dsl'] = dat['r_DSL'][0][i:i+K]
                params['dsf'] = dat['r_DSF'][0][i:i+K]
        
        if self.gendat == True:
            
            i = np.random.randint(0, len(dat)-1)
            params = dat[i].copy()
            params['const_noise'] = self.const_noise*np.random.randn(2*K)
            
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.

        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.

        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()

        Returns:
            tuple:

            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.

        """
        in_noise = self.in_noise
        onset = self.T / 4.0 # T is trial length in ms, onset is at T/4
        stim_dur = self.T / 2.0
        

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        if self.nDist == False:
            dsl = params['dsl']
            dsf = params['dsf']
        else:
            dsl = params['ndsl']
            dsf = params['ndsf']
            
        choice = params['choice']
        correct = params['correct']
        K = self.K
        const_noise = params['const_noise']
        # ----------------------------------
        # Compute values
        # ----------------------------------
        
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[4*i] += dsl[i] + const_noise[2*i]
            x_t[4*i+1] += dsf[i] + const_noise[2*i+1]
            x_t[4*i+2] += choice[i]
            if choice[i] == correct[i]:
                x_t[4*i+3] += 1
                    
        if onset < time < onset + stim_dur: #current trial parameters
            x_t[-2] += dsl[-1] + const_noise[-2]
            x_t[-1] += dsf[-1] + const_noise[-1]

        if time > onset + stim_dur :
            if self.trainChoice == True:
               y_t[int(choice[-1])-1] = 1
            else:           
                y_t[int(correct[-1])-1] = 1

        if time < onset + stim_dur + 10: # non-response period
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t
#
#
#
class StimHist_constant2(Task):
    """
    Parameters
    ----------
    dt (float): 
        The simulation timestep.
    tau (float): The intrinsic time constant of neural state decay.
    T (float): The length of a trial.
    N_batch (int): The number of trials in a batch.
    N_in (int): The input dimensionality. Set to 4*(K-1)+2: dsl, dsf, choice, reward for each past trial; dsl, dsf for the current trial.
    N_out (int): The output dimensionality. Set to 4.
    K (int): The total number of trials to consider, including the current trial and K-1 history trials.
    dat1 (dataframe): The source data.
    dat2 (dataframe), optional): A second dataset, where one of the two datasets is chosen at random at each trial.
    pdat2 (float, optional): The probability of choosing from dataset dat2 on any given trial.
    in_noise (float, optional): The input noise parameter, to scale Gaussian noise generated at each timestep. The default is 0.01.
    vis_noise (float, optional): The visual stimulus noise parameter, to scale Gaussian noise added to visual inputs (current and past), generated at each trial. The default is 0.0.
    mem_noise (float, optional): The trial history noise parameter, to scale Gaussian noise added to previous trial inputs, generated at each trial. The default is 0.0.
    targChoice (bool, optional): True if the network should target the monkey's choice (rather than the correct choice). The default is False.
    gendat (bool, optional): True if trial parameters are to be drawn from a generated test dataset. The default is False.
    """
    def __init__(self, dt, tau, T, N_batch, K, dat1, dat1_inds=None, dat2=None, dat2_inds=None, pdat2=None, in_noise=0.1, vis_noise=0.0, mem_noise=0.0, targChoice=False, gendat=False, testall=False):
        N_in = 4*K-2
        N_out = 4
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat1 = dat1
        self.dat1_inds = dat1_inds
        self.dat2 = dat2
        self.dat2_inds = dat2_inds
        self.pdat2 = pdat2
        self.in_noise = in_noise
        self.vis_noise = vis_noise
        self.mem_noise = mem_noise
        self.targChoice = targChoice
        self.gendat = gendat
        self.testall = testall
        if testall==True:
            if (dat1_inds is None and self.N_batch != dat1.shape[0]-K+1) | (dat1_inds is not None and self.N_batch != dat1_inds.shape[0]):
                print('N_batch does not match data shape')
            
    def generate_trial_params(self, batch, trial):
        """"
        Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.

        Returns:
            dict: Dictionary of trial parameters.

        """
        if self.dat2 is not None:
            p_dat2 = self.pdat2
            
            if np.random.rand() < p_dat2:
                dat = self.dat2
                dat_inds = self.dat2_inds
            else:
                dat = self.dat1
                dat_inds = self.dat1_inds
        else:
            dat = self.dat1
            dat_inds = self.dat1_inds
        
        K = self.K
   
        # ----------------------------------
        # Define trial parameters
        # ----------------------------------
        
        if self.gendat == False:
            
            if dat_inds is None:
                if  self.testall==False:
                    i = np.random.randint(0, dat.shape[0]-K+1)
                else:
                    i = trial
                    if i>=100 and i%100==0:
                        print('trial index', i)

            elif dat_inds is not None:
                if self.testall==False:
                    i = np.random.choice(dat_inds[dat_inds < dat.shape[0]-K+1])
                else:
                    i = dat_inds[trial]
                    if i>=100 and i%100==0:
                        print('trial index', i)

            params = dict()
            params['choice'] = np.array(dat['choice'][i:i+K])
            params['correct'] = np.array(dat['correct'][i:i+K])
            params['dsf'] = np.array(dat['dsf'][i:i+K])
            params['dsl'] = np.array(dat['dsl'][i:i+K])
            params['task'] = np.array(dat['task'][i:i+K])
            
            params['m_task'] = np.array(dat['m_task'][i:i+K])
            params['switch'] = np.array(dat['switch'][i:i+K])
            params['m_switch'] = np.array(dat['m_switch'][i:i+K])     
            params['err'] = np.array(dat['err'][i:i+K])
            params['perc_err'] = np.array(dat['perc_err'][i:i+K])
            params['task_err'] = np.array(dat['task_err'][i:i+K])
            
        else:
            
            i = np.random.randint(0, len(dat))
            params = dat[i].copy()
        
        params['vis_noise'] = self.vis_noise*np.random.randn(2*K)
        params['mem_noise'] = self.mem_noise*np.random.randn(4*K-4)
        params['trial_ind'] = i
            
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.

        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.

        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()

        Returns:
            tuple:

            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.

        """
        in_noise = self.in_noise
        onset = self.T / 4.0 # T is trial length in ms, onset is at T/4
        stim_dur = self.T / 2.0
        

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------

        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        vis_noise = params['vis_noise']
        mem_noise = params['mem_noise']
        K = self.K
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        x_t[0:4*K-4] += mem_noise
        
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[4*i] += dsl[i] + vis_noise[2*i]
            x_t[4*i+1] += dsf[i] + vis_noise[2*i+1]
            x_t[4*i+2] += choice[i]
            if choice[i] == correct[i]:
                x_t[4*i+3] += 1
                    
        if onset < time < onset + stim_dur: #current trial parameters
            x_t[-2] += dsl[-1] + vis_noise[-2]
            x_t[-1] += dsf[-1] + vis_noise[-1]

        if time > onset + stim_dur :
            if self.targChoice == True:
               y_t[int(choice[-1])-1] = 1
            else:           
                y_t[int(correct[-1])-1] = 1

        if time <= onset + stim_dur: # non-response period
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t
#
#
#
class StimHist_constant3(Task): #same as StimHist_constant2 but with fixed choice input (a vector of dim 4), reduced flexibility - only for targeting choice, different indexing, and a few other minor changes
    """
    Parameters
    ----------
    dt (float): 
        The simulation timestep.
    tau (float): The intrinsic time constant of neural state decay.
    T (float): The length of a trial.
    N_batch (int): The number of trials in a batch.
    N_in (int): The input dimensionality. Set to 4*(K-1)+2: dsl, dsf, choice, reward for each past trial; dsl, dsf for the current trial.
    N_out (int): The output dimensionality. Set to 4.
    K (int): The total number of trials to consider, including the current trial and K-1 history trials.
    dat (dataframe): The source data.
    in_noise (float, optional): The input noise parameter, to scale Gaussian noise generated at each timestep. The default is 0.01.
    """
    def __init__(self, dt, tau, T, N_batch, K, dat, dat_inds, in_noise=0.1, testall=False):
        N_in = 7*(K-1)+2
        N_out = 4
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat = dat
        self.dat_inds = dat_inds
        self.in_noise = in_noise
        self.testall = testall
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
            
    def generate_trial_params(self, batch, trial):
        """"
        Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.

        Returns:
            dict: Dictionary of trial parameters.

        """
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
   
        # ----------------------------------
        # Define trial parameters
        # ----------------------------------
            
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['choice'] = np.array(dat['choice'][i-K+1:i+1])
        params['correct'] = np.array(dat['correct'][i-K+1:i+1])
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])
        params['task'] = np.array(dat['task'][i-K+1:i+1])
        
        params['m_task'] = np.array(dat['m_task'][i-K+1:i+1])
        params['switch'] = np.array(dat['switch'][i-K+1:i+1])
        params['m_switch'] = np.array(dat['m_switch'][i-K+1:i+1])     
        params['err'] = np.array(dat['err'][i-K+1:i+1])
        params['perc_err'] = np.array(dat['perc_err'][i-K+1:i+1])
        params['task_err'] = np.array(dat['task_err'][i-K+1:i+1])
        
        params['trial_ind'] = i
            
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.

        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.

        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()

        Returns:
            tuple:

            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.

        """
        in_noise = self.in_noise
        onset = self.T / 4.0 # T is trial length in ms, onset is at T/4
        stim_dur = self.T / 2.0
        

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------

        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']

        K = self.K
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[7*i] += dsl[i]
            x_t[7*i+1] += dsf[i]
            x_t[7*i+2 + choice[i]-1] += 1
            if choice[i] == correct[i]:
                x_t[7*i+6] += 1
                    
        if onset < time < onset + stim_dur: #current trial parameters
            x_t[-2] += dsl[-1]
            x_t[-1] += dsf[-1]

        if time > onset + stim_dur :
            y_t[choice[-1]-1] = 1


        if time <= onset + stim_dur: # non-response period
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t
#
#
#
class WM_TaskRule(Task):
    def __init__(self, dt=10, tau=100, T=2400, N_batch=100, in_noise=0.5, P_norule=0, P_bothrule=0, P_weakrule=0, weakstrength=0.25, fixation=False, const_rule=False):
        N_inout = 4
        if fixation==True:
            N_inout = 5
        super().__init__(N_inout, N_inout, dt, tau, T, N_batch)
        self.in_noise = in_noise
        self.P_norule = P_norule
        self.P_bothrule = P_bothrule
        self.P_weakrule = P_weakrule
        self.weakstrength = weakstrength
        self.fixation = fixation
        self.const_rule = const_rule
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        params = dict()
        a1 = rng.uniform(1,2)
        a2 = rng.uniform(1,2)
        b1 = rng.uniform(1,2)
        b2 = rng.uniform(1,2)
        task = rng.choice([0,1])
        targ = np.flatnonzero([(task==0)&(a1>a2), (task==0)&(a1<a2), (task==1)&(b1>b2), (task==1)&(b1<b2)])[0]
        
        Ps = [1-self.P_norule-self.P_bothrule, self.P_norule, self.P_bothrule]
        ruleintype = rng.choice(['reg', 'no', 'both'], p=Ps)
        taskrule = np.zeros(2)
        if ruleintype == 'no':
            pass
        elif ruleintype == 'both':
            taskrule[:2] = 1
        elif ruleintype == 'reg':
            taskrule[task] = 1
        
        if rng.uniform() < self.P_weakrule:
            taskrule *= self.weakstrength
        
        params['a1'] = a1
        params['a2'] = a2
        params['b1'] = b1
        params['b2'] = b2
        params['task'] = task
        params['targ'] = targ
        params['taskrule'] = taskrule

        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        in_noise = self.in_noise
        onset = 0 
        taskrule_dur = 500 #durations in ms
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = 400
        delay3_dur = 300
        choice_dur = 300
        
        if onset + taskrule_dur + delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = onset + taskrule_dur + delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        a1 = params['a1']
        a2 = params['a2']
        b1 = params['b1']
        b2 = params['b2']
        task = params['task']
        targ = params['targ']
        taskrule = params['taskrule']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if (((onset < time < onset + taskrule_dur) and self.const_rule==False) or self.const_rule==True): # task rule input
            x_t[:2] += taskrule
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[2] += a1
            x_t[3] += b1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[2] += a2
            x_t[3] += b2         
    
        if time > go:          
            y_t[targ] = 1
            if self.fixation:
                mask_t *= 3
    
        if time < go: # non-response period
            if self.fixation:
                x_t[4] += 1
                y_t[4] = 1
                mask_t[:-1] *= 0.1
            else:
                mask_t = np.zeros(self.N_out)
        
        return x_t, y_t, mask_t
#
#
#
class WM_TaskRule2(Task):
    def __init__(self, dt=10, tau=100, T=2400, N_batch=100, in_noise=0.5, on_rule=1, off_rule=0, const_rule=False, variable_delay=True):
        N_inout = 5 
        super().__init__(N_inout, N_inout, dt, tau, T, N_batch)
        self.in_noise = in_noise
        self.on_rule = on_rule
        self.off_rule = off_rule
        self.const_rule = const_rule
        self.variable_delay = variable_delay
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        params = dict()
        a1 = rng.uniform(1,2)
        a2 = rng.uniform(1,2)
        b1 = rng.uniform(1,2)
        b2 = rng.uniform(1,2)
        task = rng.choice([0,1])
        targ = np.flatnonzero([(task==0)&(a1>a2), (task==0)&(a1<a2), (task==1)&(b1>b2), (task==1)&(b1<b2)])[0]
        
        taskrule = np.zeros(2)
        taskrule[:] = self.off_rule 
        taskrule[task] = self.on_rule
        
        params['a1'] = a1
        params['a2'] = a2
        params['b1'] = b1
        params['b2'] = b2
        params['task'] = task
        params['targ'] = targ
        params['taskrule'] = taskrule

        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        in_noise = self.in_noise
        if self.variable_delay==False:
            taskrule_dur = 500 #durations in ms
            delay1_dur = 500
            stim_dur = 200
            delay2_dur = 400
            delay3_dur = 300
            choice_dur = 300
        else:
            taskrule_dur = 0
            delay1_dur = 500
            stim_dur=200
            delay3_dur = 150
            choice_dur = 300
            delay2_dur = rng.uniform(300,500)
        
        if taskrule_dur + delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = taskrule_dur + delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        a1 = params['a1']
        a2 = params['a2']
        b1 = params['b1']
        b2 = params['b2']
        targ = params['targ']
        taskrule = params['taskrule']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if (((onset < time < onset + taskrule_dur) and self.const_rule==False) or self.const_rule==True): # task rule input
            x_t[:2] += taskrule
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[2] += a1
            x_t[3] += b1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[2] += a2
            x_t[3] += b2         
    
        if time > go:          
            y_t[targ] = 1
            mask_t *= 3
    
        if time < go: # fixation period
            x_t[4] += 1
            y_t[4] = 1
            mask_t[:-1] *= 0.1
        
        return x_t, y_t, mask_t
#
#
#
class WM_ContextSignal(Task):
    def __init__(self, dat, dat_inds, dt=10, tau=100, T=2400, N_batch=100, K=10, in_noise=0.4, testall=False):
        N_in = 7*(K-1)+3
        N_out = 5
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat = dat
        self.dat_inds = dat_inds
        self.in_noise = in_noise
        self.testall = testall
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
                
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['choice'] = np.array(dat['choice'][i-K+1:i+1])
        params['correct'] = np.array(dat['correct'][i-K+1:i+1])
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/3
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/3
        params['task'] = np.array(dat['task'][i-K+1:i+1])
        
        params['m_task'] = np.array(dat['m_task'][i-K+1:i+1])
        params['switch'] = np.array(dat['switch'][i-K+1:i+1])
        params['m_switch'] = np.array(dat['m_switch'][i-K+1:i+1])     
        params['err'] = np.array(dat['err'][i-K+1:i+1])
        params['perc_err'] = np.array(dat['perc_err'][i-K+1:i+1])
        params['task_err'] = np.array(dat['task_err'][i-K+1:i+1])
        
        params['trial_ind'] = i

        dsf = params['dsf'][-1]
        if dsf < 0:
            params['sf1'] = rng.uniform(1+np.abs(dsf),2)
        else:
            params['sf1'] = rng.uniform(1,2-dsf)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        if dsl < 0:
            params['sl1'] = rng.uniform(1+np.abs(dsl),2)
        else:
            params['sl1'] = rng.uniform(1,2-dsl)
        params['sl2'] = params['sl1'] + dsl

        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        
        in_noise = self.in_noise
        onset = 0 
        taskrule_dur = 500 #durations in ms
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = 400
        delay3_dur = 300
        choice_dur = 300
        
        if onset + taskrule_dur + delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = onset + taskrule_dur + delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        sf1 = params['sf1']
        sf2 = params['sf2']
        sl1 = params['sl1']
        sl2 = params['sl2']
        
        K = self.K
        # ----------------------------------
        # Compute values
        # ----------------------------------
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[7*i] += dsl[i]
            x_t[7*i+1] += dsf[i]
            x_t[7*i+2 + choice[i]-1] += 1
            if choice[i] == correct[i]:
                x_t[7*i+6] += 1
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[-3] += sf1
            x_t[-2] += sl1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[-3] += sf2
            x_t[-2] += sl2        
    
        if time > go:          
            y_t[correct[-1]-1] = 1
            mask_t *= 3
    
        if time < go: # fixation period
            x_t[-1] += 1
            y_t[-1] = 1
            mask_t[:-1] *= 0.1
        
        return x_t, y_t, mask_t
#
#
#
class WM_TaskRule_varDelay(Task):
    #constant rule and variable delay
    def __init__(self, dt=10, tau=100, T=1800, N_batch=100, in_noise=0.7, on_rule=1, off_rule=0, varDelay=True):
        N_inout = 5 
        super().__init__(N_inout, N_inout, dt, tau, T, N_batch)
        self.in_noise = in_noise
        self.on_rule = on_rule
        self.off_rule = off_rule
        self.varDelay = varDelay
        
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        params = dict()
        a1 = rng.uniform(1,2)
        a2 = rng.uniform(1,2)
        b1 = rng.uniform(1,2)
        b2 = rng.uniform(1,2)
        task = rng.choice([0,1])
        targ = np.flatnonzero([(task==0)&(a1>a2), (task==0)&(a1<a2), (task==1)&(b1>b2), (task==1)&(b1<b2)])[0]
        
        taskrule = np.zeros(2)
        taskrule[:] = self.off_rule 
        taskrule[task] = self.on_rule
        
        params['a1'] = a1
        params['a2'] = a2
        params['b1'] = b1
        params['b2'] = b2
        params['task'] = task
        params['targ'] = targ
        params['taskrule'] = taskrule
        if self.varDelay==True:
            params['delay2_dur'] = rng.uniform(300,500)
        else:
            params['delay2_dur'] = 400

        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        in_noise = self.in_noise
        delay1_dur = 500
        stim_dur=200
        delay2_dur = params['delay2_dur']
        delay3_dur = 150
        choice_dur = 750 - delay2_dur
        
        if delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        a1 = params['a1']
        a2 = params['a2']
        b1 = params['b1']
        b2 = params['b2']
        targ = params['targ']
        taskrule = params['taskrule']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        x_t[:2] += taskrule
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[2] += a1
            x_t[3] += b1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[2] += a2
            x_t[3] += b2         
    
        if time > go:          
            y_t[targ] = 1
            mask_t *= 3
    
        if time < go: # fixation period
            x_t[4] += 1
            y_t[4] = 1
            mask_t[:-1] *= 0.1
        
        return x_t, y_t, mask_t
#
#
#
class WM_ContextSignal_varDelay(Task):
    def __init__(self, dat, dat_inds, dt=10, tau=100, T=1800, N_batch=100, K=10, in_noise=0.7, mem_noise=0.2, \
                 testall=False, varDelay=True, trainChoice=False, taskMask=False):
        N_in = 7*(K-1)+3
        N_out = 5
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat = dat
        self.dat_inds = dat_inds
        self.in_noise = in_noise
        self.mem_noise = mem_noise
        self.testall = testall
        self.trainChoice = trainChoice
        self.varDelay = varDelay
        self.taskMask = taskMask
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
                
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['choice'] = np.array(dat['choice'][i-K+1:i+1])
        params['correct'] = np.array(dat['correct'][i-K+1:i+1])
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/3
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/3
        params['task'] = np.array(dat['task'][i-K+1:i+1])
        if self.taskMask==True:
            params['m_task'] = np.array(dat['m_task'][i-K+1:i+1])
        
        params['trial_ind'] = i

        dsf = params['dsf'][-1]
        if dsf < 0:
            params['sf1'] = rng.uniform(1+np.abs(dsf),2)
        else:
            params['sf1'] = rng.uniform(1,2-dsf)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        if dsl < 0:
            params['sl1'] = rng.uniform(1+np.abs(dsl),2)
        else:
            params['sl1'] = rng.uniform(1,2-dsl)
        params['sl2'] = params['sl1'] + dsl
        
        if self.varDelay==True:
            params['delay2_dur'] = rng.uniform(300,500)
        else:
            params['delay2_dur'] = 400
            
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        
        in_noise = self.in_noise
        mem_noise = self.mem_noise
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = params['delay2_dur']
        delay3_dur = 150
        choice_dur = 750 - delay2_dur
        
        if delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        K = self.K
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        x_t[:7*(K-1)] += np.sqrt(2 * self.alpha * mem_noise * mem_noise) * rng.standard_normal(7*(K-1))
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        sf1 = params['sf1']
        sf2 = params['sf2']
        sl1 = params['sl1']
        sl2 = params['sl2']
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[7*i] += dsl[i]
            x_t[7*i+1] += dsf[i]
            x_t[7*i+2 + choice[i]-1] += 1
            if choice[i] == correct[i]:
                x_t[7*i+6] += 1
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[-3] += sf1
            x_t[-2] += sl1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[-3] += sf2
            x_t[-2] += sl2        
    
        if time > go:
            if self.trainChoice==False:
                y_t[correct[-1]-1] = 1
            else:
                if self.taskMask==True:
                    if params['m_task'][-1] != params['m_task'][-2]:
                        mask_t *= 1.5
                    if params['m_task'][-1] == 2:
                        y_t[:2] += 0.1
                    elif params['m_task'][-1] == 1:
                        y_t[2:] += 0.1
                    
                y_t[choice[-1]-1] = 1
                

            mask_t *= 3
    
        if time < go: # fixation period
            x_t[-1] += 1
            y_t[-1] = 1
            mask_t[:-1] *= 0.1
        
        return x_t, y_t, mask_t
#
#
#
class WM_InferTask_MTT(Task):
    def __init__(self, dat, dat_inds, dt=10, tau=100, T=1800, N_batch=100, K=10, in_noise=0.7, mem_noise=0.2, testall=False, varDelay=True, trainChoice=True):
        N_in = 7*(K-1)+3
        N_out = 5
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat = dat
        self.dat_inds = dat_inds
        self.in_noise = in_noise
        self.mem_noise = mem_noise
        self.testall = testall
        self.trainChoice = trainChoice
        self.varDelay = varDelay
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
                
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['choice'] = np.array(dat['choice'][i-K+1:i+1])
        params['correct'] = np.array(dat['correct'][i-K+1:i+1])
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/3
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/3
        params['task'] = np.array(dat['task'][i-K+1:i+1])
        
        params['trial_ind'] = i

        dsf = params['dsf'][-1]
        if dsf < 0:
            params['sf1'] = rng.uniform(1+np.abs(dsf),2)
        else:
            params['sf1'] = rng.uniform(1,2-dsf)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        if dsl < 0:
            params['sl1'] = rng.uniform(1+np.abs(dsl),2)
        else:
            params['sl1'] = rng.uniform(1,2-dsl)
        params['sl2'] = params['sl1'] + dsl
        
        if self.varDelay==True:
            params['delay2_dur'] = rng.uniform(300,500)
        else:
            params['delay2_dur'] = 400
            
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        
        in_noise = self.in_noise
        mem_noise = self.mem_noise
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = params['delay2_dur']
        delay3_dur = 150
        choice_dur = 750 - delay2_dur
        
        if delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        K = self.K
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2 * self.alpha * in_noise * in_noise) * rng.standard_normal(self.N_in)
        x_t[:7*(K-1)] += np.sqrt(2 * self.alpha * mem_noise * mem_noise) * rng.standard_normal(7*(K-1))
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        sf1 = params['sf1']
        sf2 = params['sf2']
        sl1 = params['sl1']
        sl2 = params['sl2']
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        for i in range(0, K-1): #constant trial history inputs for the duration of the trial
            x_t[7*i] += dsl[i]
            x_t[7*i+1] += dsf[i]
            x_t[7*i+2 + choice[i]-1] += 1
            if choice[i] == correct[i]:
                x_t[7*i+6] += 1
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[-3] += sf1
            x_t[-2] += sl1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[-3] += sf2
            x_t[-2] += sl2        
    
        if time > go:
            if self.trainChoice==False:
                y_t[correct[-1]-1] = 1
            else:
                y_t[choice[-1]-1] = 1
            mask_t *= 3
    
        if time < go: # fixation period
            x_t[-1] += 1
            y_t[-1] = 1
            mask_t[:-1] *= 0.1
        
        return x_t, y_t, mask_t
