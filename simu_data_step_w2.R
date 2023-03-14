# Simulate some data
library(evir)
library(actuar)
#
# n <- 1000
# set.seed(123456)
# X <- runif(n = n, 0, 1)
# x <- seq(0, 1, l = 500)
#
# sigma_tan_h=20
#
# Y <- rgpd(n = n, mu = 0,beta = 1 + 0.25*tanh(sigma_tan_h*(X-0.25)) + 0.25*tanh(sigma_tan_h*(X-0.75)),xi = 1 + 0.25*tanh(sigma_tan_h*(X-0.25)) + 0.25*tanh(sigma_tan_h*(X-0.75)) )
# par(mfrow=c(1,2))
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Sigma")
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Gamma")
#
# data_sample=data.frame(X=X,Y=Y)
#
# data_test=data.frame(X=x)


# simulate_data=function(n=1000,seed=1,sigma_tan_h=20){
#
#   set.seed(seed)
#
#   X <- runif(n = n, 0, 1)
#
#   Y <- rgpd(n = n, mu = 0,beta = 1 + 0.25*tanh(sigma_tan_h*(X-0.25)) + 0.25*tanh(sigma_tan_h*(X-0.75)),xi = 1 + 0.25*tanh(sigma_tan_h*(X-0.25)) + 0.25*tanh(sigma_tan_h*(X-0.75)) )
#
#   data_sample=data.frame(X=X,Y=Y)
#
#   return(data_sample)
# }


step_simulate_data=function(x){
  res=sign(x)
  res[res==0]=1
  return(res)
}

step_w1=function(X){
  for (i in 1:length(X)){
    if (X[i]>0.7){
     X[i]<-0.2 
    } else if (X[i]<=0.7 & X[i]>0.3){
      X[i]<-0.4
    }else {
      X[i]<-0.8
  }
  }
  return(X)
}

simulate_data_burr=function(n=100,seed=1){

  set.seed(seed)
  X <- runif(n = 10*n, 0, 1)
  Y <- rburr(n = 10*n, shape2=1/step_w1(X),shape1=1,scale = (1/(step_w1(X)/(2^step_w1(X)-1))/10))
  
  #Marche mieux a priori mais ne colle pas à l'objectif
  #Y <- rburr(n = 10*n, shape2=1/step_w1(X),shape1=1,scale = (1-step_w1(X)) )
  excesses <- Y>sort(Y)[9*n]

  data_sample=data.frame(X=X[excesses],Y=(Y[excesses]-sort(Y)[9*n]))

  return(data_sample)
}
#
# perf_analysis=function(sigma_tan_h,learning_method,learning_model,num_eval){
#   x <- seq(0, 1, l = num_eval)
#   sigma_true = 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75))
#   gamma_true = 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75))
#
#   parameters_true=cbind(sigma_true,gamma_true)
#
#   data_test=data.frame(X=x)
#
#   #Prediction
#
#   if(learning_method=="GPD"){
#     parameters_estimated=matrix(learning_model,nrow=nrow(data_test),ncol=2,byrow=TRUE)
#   }else{
#     if(learning_method=="CART"){
#       parameters_estimated=pred_CART_GPD(tree=learning_model,data_test=data_test)
#     }else{
#       if(learning_method=="Kernel"){
#         parameters_estimated=learning_model
#       }else{
#         if(learning_method=="GAM"){
#           parameters_estimated=pred_GAM(fit=learning_model,data_test=data_test)
#         }else{print("Learning method not allowed")
#           return(NULL)}
#       }
#     }
#   }
#
#   #Analysis
#
#   res_sigma=c(mean((parameters_estimated[,1]-parameters_true[,1])^2),
#               median((parameters_estimated[,1]-parameters_true[,1])^2),
#               mean(abs(parameters_estimated[,1]-parameters_true[,1])),
#               median(abs(parameters_estimated[,1]-parameters_true[,1])),
#               max(abs(parameters_estimated[,1]-parameters_true[,1])))
#
#   res_gamma=c(mean((parameters_estimated[,2]-parameters_true[,2])^2),
#               median((parameters_estimated[,2]-parameters_true[,2])^2),
#               mean(abs(parameters_estimated[,2]-parameters_true[,2])),
#               median(abs(parameters_estimated[,2]-parameters_true[,2])),
#               max(abs(parameters_estimated[,2]-parameters_true[,2])))
#
#   #All
#   Y_estimated=matrix(nrow=1000,ncol=length(x))
#   Y_true=matrix(nrow=1000,ncol=length(x))
#
#   set.seed(1)
#
#   for(col in 1:length(x)){
#     Y_estimated[,col] <- rgpd(n = 1000, mu = 0,beta = parameters_estimated[col,1],xi = parameters_estimated[col,2])
#     Y_true[,col] <- rgpd(n = 1000, mu = 0,beta = parameters_true[col,1],xi = parameters_true[col,2])
#   }
#
#
#   res_all=c((quantile(Y_estimated,probs=0.5)-quantile(Y_true,probs=0.5))/quantile(Y_true,probs=0.5),
#             (quantile(Y_estimated,probs=0.75)-quantile(Y_true,probs=0.75))/quantile(Y_true,probs=0.75),
#             (quantile(Y_estimated,probs=0.9)-quantile(Y_true,probs=0.9))/quantile(Y_true,probs=0.9),
#             (quantile(Y_estimated,probs=0.95)-quantile(Y_true,probs=0.95))/quantile(Y_true,probs=0.95))
#
#   return(list(res_sigma,res_gamma,res_all))
#
# }
#


#
#
# perf_analysis_burr=function(sigma_tan_h,learning_method,learning_model,num_eval,data_sample){
#
#   if(!is.na(num_eval)){
#     #We evaluate the model on num_eval points over [0;1]
#     x <- seq(0, 1, l = num_eval)
#   }else{
#     #We evaluate the model on the train data
#     x=data_sample$X
#   }
#
#   if(sigma_tan_h<+Inf){
#     gamma_true = 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75))
#   }else{
#     gamma_true = 1 + 0.25*step_simulate_data(x-0.25) + 0.25*step_simulate_data(x-0.75)
#   }
#
#   data_test=data.frame(X=x)
#
#   #Prediction
#
#   if(learning_method=="GPD"){
#     parameters_estimated=matrix(learning_model,nrow=nrow(data_test),ncol=2,byrow=TRUE)
#   }else{
#     if(learning_method=="CART"){
#       parameters_estimated=pred_CART_GPD(tree=learning_model,data_test=data_test)
#     }else{
#       if(learning_method=="Kernel"){
#         parameters_estimated=learning_model
#       }else{
#         if(learning_method=="GAM"){
#           parameters_estimated=pred_GAM(fit=learning_model,data_test=data_test)
#         }else{print("Learning method not allowed")
#           return(NULL)}
#       }
#     }
#   }
#
#
#   #Analysis
#
#   res_gamma=c(mean((parameters_estimated[,2]-gamma_true)^2),
#               median((parameters_estimated[,2]-gamma_true)^2),
#               mean(abs(parameters_estimated[,2]-gamma_true)),
#               median(abs(parameters_estimated[,2]-gamma_true)),
#               max(abs(parameters_estimated[,2]-gamma_true)))
#
#   # res_gamma=c(mean((parameters_estimated[,2]/gamma_true - 1)^2),
#   #             median((parameters_estimated[,2]/gamma_true - 1)^2),
#   #             mean(abs(parameters_estimated[,2]/gamma_true - 1)),
#   #             median(abs(parameters_estimated[,2]/gamma_true - 1)),
#   #             max(abs(parameters_estimated[,2]/gamma_true -1)))
#
#
#   #All
#   # Y_estimated=matrix(nrow=1000,ncol=length(x))
#   # Y_true=matrix(nrow=1000,ncol=length(x))
#   # LL_res_true_par_estimated=double(length(x))
#   # LL_res_estimated_par_estimated=double(length(x))
#   #
#   # for(col in 1:length(x)){
#   #   set.seed(1)
#   #   Y_estimated[,col] <- rgpd(n = 1000, mu = 0,beta = parameters_estimated[col,1],xi = parameters_estimated[col,2])
#   #   set.seed(1)
#   #   Y_temp <- rburr(n = 10*1000, shape1=1,shape2 = 1/gamma_true[col],scale = 1)
#   #   excesses <- Y_temp>sort(Y_temp)[9*1000]
#   #   Y_true[,col]  <- Y_temp[excesses]-sort(Y_temp)[9*1000]
#   #
#   #   # LL_true[col]=sum(log(dgpd(x = Y_true[,col],
#   #   #             mu = 0,
#   #   #             beta = parameters_estimated[col,1],
#   #   #             xi = parameters_estimated[col,2])))
#   #
#   #   LL_tmp=log(dgpd(x = Y_estimated[,col],
#   #                         mu = 0,
#   #                         beta = parameters_estimated[col,1],
#   #                         xi = parameters_estimated[col,2]))
#   #   LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])
#   #
#   #   LL_res_estimated_par_estimated[col]=sum(LL_tmp)/length(Y_estimated[,col])
#   #
#   #   LL_tmp=log(dgpd(x = Y_true[,col],
#   #                         mu = 0,
#   #                         beta = parameters_estimated[col,1],
#   #                         xi = parameters_estimated[col,2]))
#   #   LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])
#   #
#   #   LL_res_true_par_estimated[col]=sum(LL_tmp)/length(Y_true[,col])
#   #
#   # }
#
# ##########################
#
#
#     # set.seed(1)
#     # x <- seq(0, 1, l = num_eval)
#     # Y <- list()
#     # LL_res_true_par_estimated=double(length(x))
#     # for(col in 1:length(x)){
#     #   if(sigma_tan_h<+Inf){
#     #     Y[[col]] <- rburr(n = 10000, shape1=1,shape2 = 1/(1 + 0.25*tanh(sigma_tan_h*(x[col]-0.25)) + 0.25*tanh(sigma_tan_h*(x[col]-0.75))),scale = 1/10)
#     #   }else{
#     #     Y[[col]] <- rburr(n = 10000, shape1=1,shape2 = 1/(1 + 0.25*step_simulate_data(x[col]-0.25) + 0.25*step_simulate_data(x[col]-0.75)),scale = 1/10)
#     #   }
#     # }
#     # threshold=sort(unlist(Y))[900000]
#     # excesses <- list()
#     # for(col in 1:length(x)){
#     #
#     #   if(sum(Y[[col]]>threshold)>0){
#     #
#     #     excesses[[col]] <- Y[[col]][Y[[col]]>threshold] - threshold
#     #
#     #     LL_tmp=log(dgpd(x = excesses[[col]],
#     #                         mu = 0,
#     #                         beta = parameters_estimated[col,1],
#     #                         xi = parameters_estimated[col,2]))
#     #     LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])
#     #
#     #     LL_res_true_par_estimated[col]=sum(LL_tmp)
#     #
#     # }else{LL_res_true_par_estimated[col]=NA}
#     #
#     # }
#
#
#     ##########################
#
#     data_sample_perf=simulate_data_burr(n=100000,seed=1,sigma_tan_h=sigma_tan_h)
#     Y <- list()
#     x <- seq(0, 1, l = num_eval)
#     LL_res_true_par_estimated=double(length(x))
#     for(col in 1:(num_eval-1)){
#       Y[[col]]<-data_sample_perf$Y[data_sample_perf$X>x[col] & data_sample_perf$X<=x[col+1]]
#     }
#     for(col in 1:(length(x)-1)){
#       if(length(Y[[col]])>0){
#
#         LL_tmp=log(dgpd(x = Y[[col]],
#                             mu = 0,
#                             beta = parameters_estimated[col,1],
#                             xi = parameters_estimated[col,2]))
#         LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])
#
#         LL_res_true_par_estimated[col]=sum(LL_tmp)
#
#     }else{LL_res_true_par_estimated[col]=NA}
#
#     }
#
#
#   # res_all=c((quantile(Y_estimated,probs=0.5)-quantile(Y_true,probs=0.5))/quantile(Y_true,probs=0.5),
#   #           (quantile(Y_estimated,probs=0.75)-quantile(Y_true,probs=0.75))/quantile(Y_true,probs=0.75),
#   #           (quantile(Y_estimated,probs=0.9)-quantile(Y_true,probs=0.9))/quantile(Y_true,probs=0.9),
#   #           (quantile(Y_estimated,probs=0.95)-quantile(Y_true,probs=0.95))/quantile(Y_true,probs=0.95))
#
#   res_all=c(1,sum(LL_res_true_par_estimated,na.rm=TRUE)/sum(!is.na(LL_res_true_par_estimated)))
#
#   return(list(res_gamma,res_gamma,res_all))
#
# }
#
# # GPD ---------------------------------------------------------------------
# #
# # GPD=evir::gpd(Y, threshold = 0, method = "ml")
# # GPD$par.ests
#
# #V0
#
# #Critères de comparaison
#
# #Quantiles
#
# #Erreur quadratique
# #Vraissemblance sur plus de données simulées
#
# #Tester différentes dépendances (sigma / gamma)
#
# #V1
#
# #Notion d'intervalles de confiance (à voir après)
#
# #Comprendre un peu plus les GAM et leurs applications
#
# #Implémenter le Single Index pour comparer
#
# #Notion de test
