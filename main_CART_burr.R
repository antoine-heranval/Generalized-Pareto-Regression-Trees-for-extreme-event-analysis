source("./simu_experience.R")
source("./simu_cart.R")

###CHOICE

#source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu.R")
source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu_big_data.R")

###CHOICE

source("./simu_data.R")
#source("./simu_data_new_design.R")

start_experience=1
nb_experience=2
num_eval=100 #or NA
sigma_tan_h_1=10
sigma_tan_h_2=+Inf #or +Inf
#
# exp_CART_10_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_50_burr,"./exp_CART_10_50_burr.csv")
# write.csv2(exp_CART_50_50_burr,"./exp_CART_50_50_burr.csv")
#
# exp_CART_10_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
#
# write.csv2(exp_CART_10_100_burr,"./exp_CART_10_100_burr.csv")
# write.csv2(exp_CART_50_100_burr,"./exp_CART_50_100_burr.csv")
#
#
# exp_CART_10_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_250_burr,"./exp_CART_10_250_burr.csv")
# write.csv2(exp_CART_50_250_burr,"./exp_CART_50_250_burr.csv")

# exp_CART_10_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_500_burr,"./exp_CART_10_500_burr.csv")
# write.csv2(exp_CART_50_500_burr,"./exp_CART_50_500_burr.csv")
#
#
# exp_CART_10_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_1000_burr,"./exp_CART_10_1000_burr.csv")
# write.csv2(exp_CART_50_1000_burr,"./exp_CART_50_1000_burr.csv")

#
# exp_CART_10_2500_burr=experience_burr(n=2500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
experience_burr(n=100,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)

# write.csv2(exp_CART_10_2500_burr,"./exp_CART_10_2500_burr.csv")
# write.csv2(exp_CART_50_2500_burr,"./exp_CART_50_2500_burr.csv")
#
# exp_CART_10_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_5000_burr,"./exp_CART_10_5000_burr.csv")
# write.csv2(exp_CART_50_5000_burr,"./exp_CART_50_5000_burr.csv")
#
# exp_CART_10_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_10000_burr,"./exp_CART_10_10000_burr.csv")
# write.csv2(exp_CART_50_10000_burr,"./exp_CART_50_10000_burr.csv")


save.image(file="./main_CART_burr_2.RData")
