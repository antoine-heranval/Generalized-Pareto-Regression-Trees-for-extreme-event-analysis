source("./simu_experience.R")
source("./simu_data.R")
#source("./simu_data_new_design.R")

source("./simu_QRM.R")
source("./simu_cart.R")
source("./simu_kernel_0.R")

back_perf_experience_burr=function(){

  list_method=c("CART","GAM","Kernel")
  list_n=c(500,1000,2500)
  list_sigma_h=c(10,Inf)

  for(method in list_method){
    for(n in list_n){
      for(sigma_h in list_sigma_h){

        assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",method,"_",sigma_h,"_",n,"_burr.rds")))


        perf_sigma=matrix(nrow=length(save_learning_model),ncol=5)
        perf_gamma=matrix(nrow=length(save_learning_model),ncol=5)
        perf_all=matrix(nrow=length(save_learning_model),ncol=2)


        for(exp in 1:length(save_learning_model)){

              learning_model=save_learning_model[[exp]]

              perf=perf_analysis_burr(sigma_h,method,learning_model,num_eval=100,data_sample=NULL)

              perf_sigma[exp,]=perf[[1]]
              perf_gamma[exp,]=perf[[2]]
              perf_all[exp,]=perf[[3]]

        }

        write.csv2(perf_gamma,paste0("./MSE_exp_",method,"_",sigma_h,"_",n,"_burr.csv"))
        write.csv2(perf_all,paste0("./LL_exp_",method,"_",sigma_h,"_",n,"_burr.csv"))
        # write.csv2(list(perf_sigma,perf_gamma,perf_all),paste0("./exp_",method,"_",sigma_h,"_",n,"_burr.csv"))

      }
    }
  }
}



source("./simu_experience.R")
source("./simu_data.R")

source("./simu_QRM.R")
source("./simu_cart.R")
source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu_big_data.R")

source("./simu_kernel_0.R")

back_mean_experience_burr=function(){

  list_method=c("GPD","CART","GAM","Kernel")
  #list_n=c(50,100,250,500)
  list_n=c(5000)
  list_sigma_h=c(Inf)
  num_eval=100

  for(method in list_method){
    for(n in list_n){
      for(sigma_h in list_sigma_h){

        x <- seq(0, 1, l = num_eval)

        if(sigma_h<+Inf){
          gamma_true = 1 + 0.25*tanh(sigma_h*(x-0.25)) + 0.25*tanh(sigma_h*(x-0.75))
        }else{
          gamma_true = 1 + 0.25*step_simulate_data(x-0.25) + 0.25*step_simulate_data(x-0.75)
        }

        data_test=data.frame(X=x)

        assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",method,"_",sigma_h,"_",n,"_burr.rds")))

        sigma=matrix(nrow=length(x),ncol=length(save_learning_model))
        gamma=matrix(nrow=length(x),ncol=length(save_learning_model))
        LL=matrix(nrow=length(x),ncol=length(save_learning_model))

        set.seed(1)
        x <- seq(0, 1, l = num_eval)
        Y <- list()
        for(col in 1:length(x)){
          if(sigma_h<+Inf){
            Y[[col]] <- rburr(n = 10000, shape1=1,shape2 = 1/(1 + 0.25*tanh(sigma_h*(x[col]-0.25)) + 0.25*tanh(sigma_h*(x[col]-0.75))),scale = 1/10)
          }else{
            Y[[col]] <- rburr(n = 10000, shape1=1,shape2 = 1/(1 + 0.25*step_simulate_data(x[col]-0.25) + 0.25*step_simulate_data(x[col]-0.75)),scale = 1/10)
          }
        }
        threshold=sort(unlist(Y))[900000]
        excesses <- list()
        for(col in 1:length(x)){
          if(sum(Y[[col]]>threshold)>0){
            excesses[[col]] <- Y[[col]][Y[[col]]>threshold] - threshold
          }else{
            excesses[[col]] <- NA
          }
        }

        for(exp in 1:length(save_learning_model)){

              learning_model=save_learning_model[[exp]]

                #Prediction

                if(method=="GPD"){
                  parameters_estimated=matrix(learning_model,nrow=nrow(data_test),ncol=2,byrow=TRUE)
                }else{
                  if(method=="CART"){
                    parameters_estimated=pred_CART_GPD(tree=learning_model,data_test=data_test)
                  }else{
                    if(method=="Kernel"){
                      parameters_estimated=learning_model
                    }else{
                      if(method=="GAM"){
                        parameters_estimated=pred_GAM(fit=learning_model,data_test=data_test)
                      }else{print("Learning method not allowed")
                        return(NULL)}
                    }
                  }
                }
            sigma[,exp]=parameters_estimated[,1]
            gamma[,exp]=parameters_estimated[,2]

            #Log-likelihoood

            for(col in 1:length(x)){

              if(sum(Y[[col]]>threshold)>0){

                LL_tmp=log(dgpd(x = excesses[[col]],
                                    mu = 0,
                                    beta = parameters_estimated[col,1],
                                    xi = parameters_estimated[col,2]))
                LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])

                LL[col,exp]=sum(LL_tmp)/length(excesses[[col]])

            }else{LL_res_true_par_estimated[col]=NA}

            }


        }

        n_MSE=sqrt(colSums(((gamma/matrix(gamma_true,nrow=length(x),ncol=length(save_learning_model),byrow=FALSE))-1)^2))
        mean_gamma=gamma[,which.min(abs(n_MSE-median(n_MSE)))]
        #mean_gamma=gamma[,which.min(n_MSE)]

        mean_sigma=rowMeans(sigma,na.rm=TRUE)
        # mean_gamma=rowMeans(gamma,na.rm=TRUE)
        mean_LL=rowMeans(LL)

        write.csv2(list(mean_sigma,mean_gamma,mean_LL),paste0("./mean_exp_",method,"_",sigma_h,"_",n,"_burr.csv"))

      }
    }
  }
}
