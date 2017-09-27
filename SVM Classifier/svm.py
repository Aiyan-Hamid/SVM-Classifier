import matplotlib.pyplot as plt
import numpy as np

class SVM:
    def __init__(self,data):
        self.data=data

        self.colors={1:'r',-1:'b'}

        optimization={}

        transform=[[1,1],[-1,1],[-1,-1],[1,-1]] #for kernel

        all_data=[]

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)


        self.max_feature_value =max(all_data)
        self.min_feature_value =min(all_data)
        all_data = None

        step_sizes= [self.max_feature_value * 0.1, #step size for x
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,
                      ]

        b_range_multiple = 2 #step size for b

        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            itr=False
            while not itr:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transform:
                        w_trans=w*transformation
                        f=1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if yi*(np.dot(w_trans,xi)+b)<1:
                                    f=0

                        if f:
                            optimization[np.linalg.norm(w_trans)]=[w_trans,b]
                if w[0]<0:
                    itr=True

                else:
                    w=w-step

            norms=sorted([n for n in optimization])
            opt_choice = optimization[norms[0]]
            self.w=opt_choice[0]
            self.b=opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))


    def predict(self,features):
        """we need to check the sign of (x.w+b). If sign is negative it is -ve side
                else if the sign is positive give data is on +ve side."""

        p=np.sign(np.dot(np.array(features),self.w)+self.b)
        if(p!=0):
            plt.scatter(features[0],features[1],s=120,marker='o',c=self.colors[p])
        return p

    
    def display(self):
        
        [[plt.scatter(x[0],x[1],s=100,marker='o',color=self.colors[i])for x in training_data[i]]for i in training_data]

        def hyperplane(x,w,b):
            q=(-w[0]*x-b)/w[1]
            return q

        
        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]


        #line to display Hyperplane
        p1=hyperplane(hyp_x_min,self.w,self.b)
        p2=hyperplane(hyp_x_max,self.w,self.b)
        plt.plot([hyp_x_min,hyp_x_max],[p1,p2],'-')
        plt.show()

        




training_data={-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}
svm=SVM(data=training_data)
#svm.fit(data=training_data)

testing_data=[[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]

for test in testing_data:
    svm.predict(test)

svm.display()
