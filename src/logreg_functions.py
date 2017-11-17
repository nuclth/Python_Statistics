# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:11:11 2017

@author: Alex
"""

# import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# user defined matplotlib plotting values
mpl.rcParams['figure.figsize'] = (7,7)
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['lines.markersize'] = 1000
mpl.rcParams['legend.fontsize'] = 30


# user defined plotting functions

def plot_roc (df):
    """Function to plot the receiver operating characteristic for our logistic
    regression. The input df is a dataframe holding our false positive rate (fpr)
    and true positive rate (tpr)."""
    plt.plot([0,1], 'k--', label = 'Random Guess')
    plt.plot(df['fpr'], df['tpr'], label = 'Log. Reg.')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.legend (loc=4, fontsize=20)
    plt.tight_layout()
    plt.show()
    
def plot_cat (df, ax=None, rel=False):
    """Function to create a bar plot of a binary variable against a categorical
    variable. Response is either in counts or relative percentage."""
    ax.set_ylabel('Counts')    
    plot_data = df.copy()
    plot_data = plot_data.groupby(['parent_smoking', 'student_smoking']).size()
        
    if rel:
        ax.set_ylabel('% of Population')
        plot_data = plot_data / plot_data.unstack(level=1).sum() * 100
    
    plot_data.unstack(level=1).plot(kind='bar', rot=0, ax=ax)
    ax.set_xlabel('A Parent Smokes')
    legend = ax.legend(title='Student Smokes', loc=0)
    legend.get_title().set_fontsize('20')
    
    
def plot_cat_mod (df, ax=None, rel=False):
    """Function to create a bar plot of a binary variable against a categorical
    variable. Response is either in counts or relative percentage."""
    ax.set_ylabel('Counts')    
    plot_data = df.copy()
    plot_data = plot_data.groupby(['parent_smoking', 'student_smoking']).size()
        
    if rel:
        ax.set_ylabel('% of Population')
        plot_data = plot_data / plot_data.unstack(level=1).sum() * 100
    
    plot_data.unstack(level=1).plot(kind='bar', rot=0, ax=ax)
    ax.set_xlabel('# of Smoking Parents')
    legend = ax.legend(title='Student Smokes', loc=0)
    legend.get_title().set_fontsize('20')   
    
# user defined dataset creation functions
    
def create_smoking_data (a, b, c, d):
    """Function to create a dataframe for our smoking data. The dataframe has two columns 
    corresponding to the smoking status of students and their parents. The inputs of the function
    are the number that: 
    a = student/parent smoke, b = parent smokes, c = student smokes, d = neither smokes.
    A dataframe for each possible combination of parent/student smoking is created, then all 4 
    dataframes are concatenated. The dataframe is then shuffled so as to be random and returned."""
    df1 = pd.concat ([pd.DataFrame ([['Y','Y']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (a)], ignore_index=True)
    df2 = pd.concat ([pd.DataFrame ([['Y','N']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (b)], ignore_index=True)
    df3 = pd.concat ([pd.DataFrame ([['N','Y']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (c)], ignore_index=True)
    df4 = pd.concat ([pd.DataFrame ([['N','N']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (d)], ignore_index=True)
    
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df
    
def create_smoking_data_mod (a, b, c, d, e, f):
    """
    a = both parents and student smokes
    b = both parents smoke, student doesn't
    c = one parent smoke and student smokes
    d = one parent smokes, student doesn't
    e = no parents smoke, student does
    f = neither parents nor student smoke
    """
    df1 = pd.concat ([pd.DataFrame ([[2,'Y']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (a)], ignore_index=True)
    df2 = pd.concat ([pd.DataFrame ([[2,'N']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (b)], ignore_index=True)
    df3 = pd.concat ([pd.DataFrame ([[1,'Y']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (c)], ignore_index=True)
    df4 = pd.concat ([pd.DataFrame ([[1,'N']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (d)], ignore_index=True)
    df5 = pd.concat ([pd.DataFrame ([[0,'Y']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (e)], ignore_index=True)
    df6 = pd.concat ([pd.DataFrame ([[0,'N']] , columns = ['parent_smoking', 'student_smoking']) 
                      for i in range (f)], ignore_index=True)
    
    df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df