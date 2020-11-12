import pandas as pd
import numpy as np
import time

def merge_rankings(fd_df,rank_file):
    rank_df = pd.read_csv(rank_file)
    rank_df = rank_df[rank_df.FFPts > 4]
    df = pd.merge(left=fd_df,right=rank_df,how='left',left_on='Nickname',right_on='Player')
    
    return df

def generate_lineup(fd_rank_df,include_player='No',include_player_pos='No'):
    
    qb_df = fd_rank_df[fd_rank_df['Position'] == 'QB']
    qb_df['Usage'] = qb_df['Pass Att']
    qb_df['Usage_z'] = (qb_df['Usage'] - qb_df['Usage'].mean())/qb_df['Usage'].std(ddof=0)
    qb_df['FFPts_z'] = (qb_df['FFPts'] - qb_df['FFPts'].mean())/qb_df['FFPts'].std(ddof=0)
    
    rb_df = fd_rank_df[fd_rank_df['Position'] == 'RB']
    rb_df['Usage'] = rb_df['Rush Att'] + rb_df['Rec']
    rb_df['Usage_z'] = (rb_df['Usage'] - rb_df['Usage'].mean())/rb_df['Usage'].std(ddof=0)
    rb_df['FFPts_z'] = (rb_df['FFPts'] - rb_df['FFPts'].mean())/rb_df['FFPts'].std(ddof=0)
    
    wr_df = fd_rank_df[fd_rank_df['Position'] == 'WR']
    wr_df['Usage'] = wr_df['Rush Att'] + wr_df['Rec']
    wr_df['Usage_z'] = (wr_df['Usage'] - wr_df['Usage'].mean())/wr_df['Usage'].std(ddof=0)
    wr_df['FFPts_z'] = (wr_df['FFPts'] - wr_df['FFPts'].mean())/wr_df['FFPts'].std(ddof=0)
    
    te_df = fd_rank_df[fd_rank_df['Position'] == 'TE']
    te_df['Usage'] = te_df['Rush Att'] + te_df['Rec']
    te_df['Usage_z'] = (te_df['Usage'] - te_df['Usage'].mean())/te_df['Usage'].std(ddof=0)
    te_df['FFPts_z'] = (te_df['FFPts'] - te_df['FFPts'].mean())/te_df['FFPts'].std(ddof=0)
    
    flx_df = fd_rank_df[fd_rank_df['Position'].isin(['RB','WR','TE'])]
    flx_df['Usage'] = flx_df['Rush Att'] + flx_df['Rec']
    flx_df['Usage_z'] = (flx_df['Usage'] - flx_df['Usage'].mean())/flx_df['Usage'].std(ddof=0)
    flx_df['FFPts_z'] = (flx_df['FFPts'] - flx_df['FFPts'].mean())/flx_df['FFPts'].std(ddof=0)
    
    def_df = fd_rank_df[fd_rank_df['Position'] == 'D']
    def_df['Usage'] = def_df['FPPG']
    def_df['Usage_z'] = (def_df['Usage'] - def_df['Usage'].mean())/def_df['Usage'].std(ddof=0)
    def_df['FFPts_z'] = (def_df['Usage'] - def_df['Usage'].mean())/def_df['Usage'].std(ddof=0)
    
    qb = 1
    rb = 2
    wr = 3
    te = 1
    flx = 1
    d = 1
    
    if include_player_pos == 'QB':
        qb_sel = qb_df[qb_df['Nickname'] == include_player]
    else:
        qb_sel = qb_df.sample(n=qb)
        
    if include_player_pos == 'RB':
        rb_sel_p = rb_df[rb_df['Nickname'] == include_player]
        rb_sel_r = rb_df.sample(n=rb-1)
        rb_sel = pd.concat([rb_sel_p,rb_sel_r])
    else:
        rb_sel = rb_df.sample(n=rb)
        
    if include_player_pos == 'WR':
        wr_sel_p = wr_df[wr_df['Nickname'] == include_player]
        wr_sel_r = wr_df.sample(n=wr-1)
        wr_sel = pd.concat([wr_sel_p,wr_sel_r])
    else:
        wr_sel = wr_df.sample(n=wr)
        
    if include_player_pos == 'TE':
        te_sel = te_df[te_df['Nickname'] == include_player]
    else:
        te_sel = te_df.sample(n=te)

    if include_player_pos == 'FLEX':
        flx_sel = flx_df[flx_df['Nickname'] == include_player]
    else:
        flx_sel = flx_df.sample(n=flx)
    
    if include_player_pos == 'DEF':
        def_sel = def_df[def_df['Nickname'] == include_player]
    else:
        def_sel = def_df.sample(n=d)
    
    qb_sel = qb_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    rb_sel = rb_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    wr_sel = wr_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    te_sel = te_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    flx_sel = flx_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    def_sel = def_sel[['Position','First Name','Last Name','Salary','Team_x','Injury Indicator','FFPts','Usage','FFPts_z','Usage_z']]
    
    lineup = pd.concat([qb_sel,rb_sel,wr_sel,te_sel,flx_sel,def_sel])
    
    return lineup

def fitness(lineup_df,function='FP'):
    salary = sum(lineup_df['Salary'])
    
    if function == 'FP':
        points = sum(lineup_df['FFPts'])
    elif function == 'Volume':
        points = sum(lineup_df['Usage_z'])    
    elif function == 'Combo':
        points = sum(lineup_df['FFPts_z'] + lineup_df['Usage_z'])
    
    if salary > 60000:
        return -1
    else:
        return points
    

def run_optimzer(fd_file,rank_file,function,player,player_pos,iterations):
    fd_df = pd.read_csv(fd_file)
    fd_rank_df = merge_rankings(fd_df,rank_file)
    fd_rank_df.loc[fd_rank_df['Position'] == 'D', 'FFPts'] = fd_rank_df['FPPG']
    fd_rank_df = fd_rank_df[fd_rank_df['FFPts'].notna()]
    fd_rank_df = fd_rank_df[fd_rank_df['Injury Indicator'] != 'IR']
    
    best_score = 0
    best_lineup = generate_lineup(fd_rank_df)
    
    for i in range(0,iterations):        
        lineup_df = generate_lineup(fd_rank_df,player,player_pos)
        score = fitness(lineup_df,function)
    
        if score > best_score:
            best_score = score
            best_lineup = lineup_df
            print("New best of " + str(best_score) + " found on iteration " + str(i))

    return best_score, best_lineup

if __name__ == "__main__":
    
    ##### Need to drop people who are on IR, and make sure the same player can't be picked in Flex
    
    fd_file = 'FanDuel-NFL-2020-11-15-51566-players-list.csv' #Fanduel player download
    rank_file = '4for4_W10_fd.csv' #4for4 ranking file for Fanduel
    iterations = 500000 #Number of generations
    
    start_time = time.time()
    function = 'FP' #FP or Volume
    player = 'No' #Player full name
    player_pos = 'No' #Position to put player in    
    score1, lineup1 = run_optimzer(fd_file,rank_file,function,player,player_pos,iterations)
    print("FP Lineup = " + str(sum(lineup1.FFPts)))
    lineup1.to_csv('fp_lineup.csv')
    print("--- %s minutes ---" % ((time.time() - start_time)*60))
    
    start_time = time.time()
    function = 'Volume' #FP or Volume
    player = 'No' #Player full name
    player_pos = 'No' #Position to put player in
    score2, lineup2 = run_optimzer(fd_file,rank_file,function,player,player_pos,iterations)
    print("Volume Lineup = " + str(sum(lineup2.FFPts)))
    lineup2.to_csv('vol_lineup.csv')
    print("--- %s minutes ---" % ((time.time() - start_time)*60))
    
    start_time = time.time()
    function = 'Combo' #FP or Volume
    player = 'No' #Player full name
    player_pos = 'No' #Position to put player in
    score3, lineup3 = run_optimzer(fd_file,rank_file,function,player,player_pos,iterations)
    print("Combo Lineup = " + str(sum(lineup3.FFPts)))
    lineup3.to_csv('combo_lineup.csv')
    print("--- %s minutes ---" % ((time.time() - start_time)*60))
    
    # function = 'FP' #FP or Volume
    # player = 'Aaron Jones' #Player full name
    # player_pos = 'RB' #Position to put player in
    # score3, lineup3 = run_optimzer(fd_file,rank_file,function,player,player_pos,iterations)
    
    # function = 'Volume' #FP or Volume
    # player = 'Aaron Jones' #Player full name
    # player_pos = 'RB' #Position to put player in
    # score4, lineup4 = run_optimzer(fd_file,rank_file,function,player,player_pos,iterations)

    

            