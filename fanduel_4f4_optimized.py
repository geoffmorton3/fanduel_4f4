import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import csv
import time

def fitness(fp,sal,pos,nm):
    qb = 1
    rb = 2
    wr = 3
    te = 1
    flx = 1
    d = 1
    
    weights = [qb,rb,wr,te,flx,d]
    
    salary = 0
    points = 0
    nm_list = []
    
    for i in range(0,len(weights)):
        lineup = (sorted(zip(pos[i], fp[i], sal[i], nm[i]), reverse=True)[:weights[i]])
        nm_list.append(lineup)
        
        for j in range(0,len(lineup)):
            points += lineup[j][1]
            salary += lineup[j][2]
            
    if salary > 60000:
        points = -1
        
    nms = []
    for i in range(0,len(nm_list)):
        for j in range(0,len(nm_list[i])):
            nms.append(nm_list[i][j][3])
    
    if len(set(nms)) != 9:
        points = -1
    
    return points

def get_lineup(fp,sal,pos,nm):
    qb = 1
    rb = 2
    wr = 3
    te = 1
    flx = 1
    d = 1
    
    weights = [qb,rb,wr,te,flx,d]
    lineup = []
    
    for i in range(0,len(weights)):
        lineup.append(sorted(zip(pos[i], fp[i], sal[i], nm[i]), reverse=True)[:weights[i]])
        
    return lineup

#--- MAIN 
class Particle:
    def __init__(self,x0,x1,x2):
        self.fp_list=x0        # original list of values
        self.sal_list=x1
        self.nm_list=x2
        self.position_i=[]         # particle position
        self.velocity_i=[]         # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,len(x0)):
            l1 = []
            l2 = []
            l3 = []
            for j in range(0,len(x0[i])):
                l1.append(random.uniform(-1,1))
                l2.append(random.uniform(-1,1))
                l3.append(random.uniform(-1,1))
                
            self.position_i.append(l1)
            self.velocity_i.append(l2)
            self.pos_best_i.append(l3)

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.fp_list,self.sal_list,self.position_i,self.nm_list)

        # check to see if the current position is an individual best
        if self.err_i > self.err_best_i:
            for i in range(0,len(self.position_i)):
                for j in range(0,len(self.position_i[i])):
                    self.pos_best_i[i][j] = self.position_i[i][j]
                    self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2       # social constant

        for i in range(0,len(pos_best_g)):
            for j in range(0,len(pos_best_g[i])):
                r1=random.random()
                r2=random.random()
                
                vel_cognitive=c1*r1*(self.pos_best_i[i][j]-self.position_i[i][j])
                vel_social=c2*r2*(pos_best_g[i][j]-self.position_i[i][j])
                self.velocity_i[i][j]=w*self.velocity_i[i][j]+vel_cognitive+vel_social
                
    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0,len(self.position_i)):
            for j in range(0,len(self.position_i[i])):
                self.position_i[i][j]=self.position_i[i][j]+self.velocity_i[i][j]
    
                # adjust maximum position if necessary
                if self.position_i[i][j]>1:
                    self.position_i[i][j]=0.9
    
                # adjust minimum position if neseccary
                if self.position_i[i][j] < -1:
                    self.position_i[i][j]=-0.9
                    
class PSO():
    def __init__(self,costFunc,x0,x1,x2,num_particles,maxiter):

        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group
        self.lineup = []

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0,x1,x2))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i > self.err_best_g:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i)
                    self.lineup = get_lineup(swarm[j].fp_list,swarm[j].sal_list,swarm[j].position_i,swarm[j].nm_list)
                    #print("After iteration " + str(i) + " score is " + str(self.err_best_g))

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position()
            i+=1
            #print([swarm[0].position_i[0][0]],[swarm[1].position_i[0][0]],[swarm[2].position_i[0][0]])


def merge_rankings(fd_df,rank_file):
    rank_df = pd.read_csv(rank_file)
    rank_df = rank_df[rank_df.FFPts > 4]
    df = pd.merge(left=fd_df,right=rank_df,how='left',left_on='Nickname',right_on='Player')
    
    return df

def get_data(metric):
    fd_file = 'FanDuel-NFL-2020-11-15-51566-players-list.csv' #Fanduel player download
    rank_file = '4for4_W10_projections.csv' #4for4 ranking file for Fanduel
    fd_df = pd.read_csv(fd_file)
    fd_rank_df = merge_rankings(fd_df,rank_file)
    fd_rank_df.loc[fd_rank_df['Position'] == 'D', 'FFPts'] = fd_rank_df['FPPG']
    fd_rank_df.loc[fd_rank_df['Position'] == 'D', 'aFPA'] = fd_rank_df['FPPG']
    fd_rank_df['Avg_Pts'] = (fd_rank_df['FPPG'] + fd_rank_df['FFPts'] + fd_rank_df['aFPA']) / 3
    fd_rank_df = fd_rank_df[fd_rank_df['FFPts'].notna()]
    fd_rank_df = fd_rank_df[fd_rank_df['Injury Indicator'] != 'IR']
    
    #team_df = fd_rank_df.groupby(by=fd_rank_df['Team_x']).sum().reset_index()
    #team_df = team_df[['Team_x','FFPts']].rename(columns={"FFPts":"Team_FFPts"})
    
    #df = pd.merge(how='left',left=fd_rank_df,right=team_df,left_on='Team_x',right_on='Team_x')
    #df['perc_team_fps'] = df['FFPts']/df['Team_FFPts']
    
    #fd_rank_df = df

    
    qb_df = fd_rank_df[fd_rank_df['Position'] == 'QB']
    qb_df['Usage'] = qb_df['Pass Att']
    qb_df['Usage_z'] = (qb_df['Usage'] - qb_df['Usage'].mean())/qb_df['Usage'].std(ddof=0)
    qb_df['FFPts_z'] = (qb_df['FFPts'] - qb_df['FFPts'].mean())/qb_df['FFPts'].std(ddof=0)
    qb_df['aFPA_z'] = (qb_df['aFPA'] - qb_df['aFPA'].mean())/qb_df['aFPA'].std(ddof=0)
    qb_df['Combo'] = (qb_df['Usage_z'] + qb_df['FFPts_z'])/2
    
    rb_df = fd_rank_df[fd_rank_df['Position'] == 'RB']
    rb_df['Usage'] = rb_df['Rush Att'] + rb_df['Rec']
    rb_df['Usage_z'] = (rb_df['Usage'] - rb_df['Usage'].mean())/rb_df['Usage'].std(ddof=0)
    rb_df['FFPts_z'] = (rb_df['FFPts'] - rb_df['FFPts'].mean())/rb_df['FFPts'].std(ddof=0)
    rb_df['aFPA_z'] = (rb_df['aFPA'] - rb_df['aFPA'].mean())/rb_df['aFPA'].std(ddof=0)
    rb_df['Combo'] = (rb_df['Usage_z'] + rb_df['FFPts_z'])/2
    
    wr_df = fd_rank_df[fd_rank_df['Position'] == 'WR']
    wr_df['Usage'] = wr_df['Rush Att'] + wr_df['Rec']
    wr_df['Usage_z'] = (wr_df['Usage'] - wr_df['Usage'].mean())/wr_df['Usage'].std(ddof=0)
    wr_df['FFPts_z'] = (wr_df['FFPts'] - wr_df['FFPts'].mean())/wr_df['FFPts'].std(ddof=0)
    wr_df['aFPA_z'] = (wr_df['aFPA'] - wr_df['aFPA'].mean())/wr_df['aFPA'].std(ddof=0)
    wr_df['Combo'] = (wr_df['Usage_z'] + wr_df['FFPts_z'])/2
    
    te_df = fd_rank_df[fd_rank_df['Position'] == 'TE']
    te_df['Usage'] = te_df['Rush Att'] + te_df['Rec']
    te_df['Usage_z'] = (te_df['Usage'] - te_df['Usage'].mean())/te_df['Usage'].std(ddof=0)
    te_df['FFPts_z'] = (te_df['FFPts'] - te_df['FFPts'].mean())/te_df['FFPts'].std(ddof=0)
    te_df['aFPA_z'] = (te_df['aFPA'] - te_df['aFPA'].mean())/te_df['aFPA'].std(ddof=0)
    te_df['Combo'] = (te_df['Usage_z'] + te_df['FFPts_z'])/2
    
    flx_df = fd_rank_df[fd_rank_df['Position'].isin(['RB','WR','TE'])]
    flx_df['Usage'] = flx_df['Rush Att'] + flx_df['Rec']
    flx_df['Usage_z'] = (flx_df['Usage'] - flx_df['Usage'].mean())/flx_df['Usage'].std(ddof=0)
    flx_df['FFPts_z'] = (flx_df['FFPts'] - flx_df['FFPts'].mean())/flx_df['FFPts'].std(ddof=0)
    flx_df['aFPA_z'] = (flx_df['aFPA'] - flx_df['aFPA'].mean())/flx_df['aFPA'].std(ddof=0)
    flx_df['Combo'] = (flx_df['Usage_z'] + flx_df['FFPts_z'])/2
     
    def_df = fd_rank_df[fd_rank_df['Position'] == 'D']
    def_df['Usage'] = def_df['FPPG']
    def_df['Usage_z'] = (def_df['Usage'] - def_df['Usage'].mean())/def_df['Usage'].std(ddof=0)
    def_df['FFPts_z'] = (def_df['Usage'] - def_df['Usage'].mean())/def_df['Usage'].std(ddof=0)
    def_df['Combo'] = def_df['Usage_z']
    
    qb_list = qb_df[metric].tolist()
    rb_list = rb_df[metric].tolist()
    wr_list = wr_df[metric].tolist()
    te_list = te_df[metric].tolist()
    flx_list = flx_df[metric].tolist()
    def_list = def_df[metric].tolist()
    
    qb_sal = qb_df['Salary'].tolist()
    rb_sal = rb_df['Salary'].tolist()
    wr_sal = wr_df['Salary'].tolist()
    te_sal = te_df['Salary'].tolist()
    flx_sal = flx_df['Salary'].tolist()
    def_sal = def_df['Salary'].tolist()
    
    qb_nm = qb_df['Nickname'].tolist()
    rb_nm = rb_df['Nickname'].tolist()
    wr_nm = wr_df['Nickname'].tolist()
    te_nm = te_df['Nickname'].tolist()
    flx_nm = flx_df['Nickname'].tolist()
    def_nm = def_df['Nickname'].tolist()
    
    pso_list = [qb_list,rb_list,wr_list,te_list,flx_list,def_list]
    pso_sal = [qb_sal,rb_sal,wr_sal,te_sal,flx_sal,def_sal]
    pso_nm = [qb_nm,rb_nm,wr_nm,te_nm,flx_nm,def_nm]
    
    return pso_list, pso_sal, pso_nm

if __name__ == "__main__":
    
    start_time = time.time()
    pso_list, pso_sal, pso_nm = get_data('FFPts')
    
    best_score = 0
    best_lineup_1 = []
    
    for i in range (0,20):
        l = PSO(fitness,pso_list,pso_sal,pso_nm,5000,25)
        if l.err_best_g > best_score:
            best_score = l.err_best_g
            best_lineup_1 = l.lineup
            print(i)
            
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    pso_list, pso_sal, pso_nm = get_data('Usage_z')
    
    best_score = 0
    best_lineup_2 = []
    
    for i in range (0,20):
        l = PSO(fitness,pso_list,pso_sal,pso_nm,5000,25)
        if l.err_best_g > best_score:
            best_score = l.err_best_g
            best_lineup_2 = l.lineup
            print(i)
    
    
    pso_list, pso_sal, pso_nm = get_data('Combo')
    
    best_score = 0
    best_lineup_3 = []
    
    for i in range (0,20):
        l = PSO(fitness,pso_list,pso_sal,pso_nm,5000,25)
        if l.err_best_g > best_score:
            best_score = l.err_best_g
            best_lineup_3 = l.lineup
            print(i)
            

    with open('output.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='\n')
        wr.writerow('1')
        wr.writerow(best_lineup_1)
        wr.writerow('2')
        wr.writerow(best_lineup_2)
        wr.writerow('3')
        wr.writerow(best_lineup_3)

    