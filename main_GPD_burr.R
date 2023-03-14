
source("./simu_experience.R")

source("./simu_data.R")
#source("./simu_data_new_design.R")

#source("./simu_GPD.R")
library(testit)
#GPD

start_experience=1
nb_experience=100
num_eval=100 #or NA
sigma_tan_h_1=10
sigma_tan_h_2=+Inf #or +Inf
#
# exp_GPD_10_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
# exp_GPD_50_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
#
# write.csv2(exp_GPD_10_50_burr,"./exp_GPD_10_50_burr.csv")
# write.csv2(exp_GPD_50_50_burr,"./exp_GPD_50_50_burr.csv")
#
# exp_GPD_10_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
# exp_GPD_50_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
#
#
# write.csv2(exp_GPD_10_100_burr,"./exp_GPD_10_100_burr.csv")
# write.csv2(exp_GPD_50_100_burr,"./exp_GPD_50_100_burr.csv")
#
#
# exp_GPD_10_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
# exp_GPD_50_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
#
# write.csv2(exp_GPD_10_250_burr,"./exp_GPD_10_250_burr.csv")
# write.csv2(exp_GPD_50_250_burr,"./exp_GPD_50_250_burr.csv")

exp_GPD_10_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
exp_GPD_50_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)

write.csv2(exp_GPD_10_500_burr,"./exp_GPD_10_500_burr.csv")
write.csv2(exp_GPD_50_500_burr,"./exp_GPD_50_500_burr.csv")


exp_GPD_10_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
exp_GPD_50_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)

write.csv2(exp_GPD_10_1000_burr,"./exp_GPD_10_1000_burr.csv")
write.csv2(exp_GPD_50_1000_burr,"./exp_GPD_50_1000_burr.csv")

exp_GPD_10_2500_burr=experience_burr(n=2500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
exp_GPD_50_2500_burr=experience_burr(n=2500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)

write.csv2(exp_GPD_10_2500_burr,"./exp_GPD_10_2500_burr.csv")
write.csv2(exp_GPD_50_2500_burr,"./exp_GPD_50_2500_burr.csv")
#
# exp_GPD_10_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
# exp_GPD_50_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
#
# write.csv2(exp_GPD_10_5000_burr,"./exp_GPD_10_5000_burr.csv")
# write.csv2(exp_GPD_50_5000_burr,"./exp_GPD_50_5000_burr.csv")
#
# exp_GPD_10_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
# exp_GPD_50_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GPD",num_eval=num_eval)
#
# write.csv2(exp_GPD_10_10000_burr,"./exp_GPD_10_10000_burr.csv")
# write.csv2(exp_GPD_50_10000_burr,"./exp_GPD_50_10000_burr.csv")
#

save.image(file="./main_GPD_burr.RData")
