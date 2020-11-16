import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import csv
import time

def fitness(fp,sal,pos,nm):
    mvp = 1
    flx = 4
    
    weights = [mvp,flx]
    
    salary = 0
    points = 0
    nm_list = []
    
    for i in range(0,len(weights)):
        lineup = (sorted(zip(pos[i], fp[i], sal[i], nm[i]), reverse=True)[:weights[i]])
        nm_list.append(lineup)
        
        for j in range(0,len(lineup)):
            if i == 0:
                points += lineup[j][1]*1.5
            else:
                points += lineup[j][1]
            salary += lineup[j][2]
            
    if salary > 60000:
        points = -1
        
    nms = []
    for i in range(0,len(nm_list)):
        for j in range(0,len(nm_list[i])):
            nms.append(nm_list[i][j][3])
    
    if len(set(nms)) != 5:
        points = -1
    
    return points

def get_lineup(fp,sal,pos,nm):
    mvp = 1
    flx = 4
    
    weights = [mvp,flx]
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


def get_data(metric):
    fd_file = 'FanDuel-NFL-2020-11-16-51718-players-list.csv' #Fanduel player download
    rank_file = '4for4_W10_projections.csv' #4for4 ranking file for Fanduel
    fd_df = pd.read_csv(fd_file)
    rank_df = pd.read_csv(rank_file)
    rank_df = rank_df[rank_df.FFPts > 2]
    
    fd_rank_df = pd.merge(left=fd_df,right=rank_df,how='left',left_on='Nickname',right_on='Player')
    fd_rank_df = fd_rank_df[fd_rank_df['FFPts'].notna()]
    fd_rank_df = fd_rank_df[fd_rank_df['Injury Indicator'] != 'IR']
    
    mvp_df = fd_rank_df.copy()
    flx_df = fd_rank_df.copy()
    
    mvp_list = mvp_df[metric].tolist()
    flx_list = flx_df[metric].tolist()
    
    mvp_sal = mvp_df['Salary'].tolist()
    flx_sal = flx_df['Salary'].tolist()
    
    mvp_nm = mvp_df['Nickname'].tolist()
    flx_nm = flx_df['Nickname'].tolist()
    
    
    pso_list = [mvp_list,flx_list]
    pso_sal = [mvp_sal,flx_sal]
    pso_nm = [mvp_nm,flx_nm]
    
    return pso_list, pso_sal, pso_nm

if __name__ == "__main__":
    
    start_time = time.time()
    pso_list, pso_sal, pso_nm = get_data('FFPts')
    
    best_score = 0
    best_lineup_1 = []
    
    for i in range (0,5):
        l = PSO(fitness,pso_list,pso_sal,pso_nm,5000,10)
        if round(l.err_best_g,3) >= best_score:
            best_score = round(l.err_best_g,3)
            best_lineup_1 = l.lineup
            print("New best of:" + str(best_score) + " found at run:" + str(i))
            
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print("Best score:" + str(best_score))


    # with open('output.csv', 'w') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='\n')
    #     wr.writerow('1')
    #     wr.writerow(best_lineup_1)
    #     wr.writerow('2')
    #     wr.writerow(best_lineup_2)
    #     wr.writerow('3')
    #     wr.writerow(best_lineup_3)