#2 Source of functions


#source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu.R")
#The name of the method is GPRT_method_xi_in_R_plus
#OR
#source("./function_generalized_pareto_regression_tree_xi_in_R.R")
#The name of the method is GPRT_method_xi_in_R_plus

#The first method add a constrain of strict positivity to the shape parameter of the GPD
#In theory, the estimation of GPD thrgough maximum likelihood is known to be assymptotically normal fo xi > -1/2 and consistant for xi >-1

#3 Growing of the tree
#
# set.seed(1)
#
# GPRT_unpruned=rpart(data = data_sample,
#                     formula = Y ~ X,
#                     method = GPRT_method_xi_in_R_plus,
#                     control = rpart.control(cp=0,minsplit=60, minbucket=20, maxcompete=10, maxsurrogate=10),
#                     xval=0)
#
# save(GPRT_unpruned,file="unpruned_GPRT.RData")
#
# #4 Pruning of the tree
#
# #a) By cross validation
#
# alpha=cross_val_gprt(data = data_sample,
#                      formula = Y ~ X,
#                      arbre = GPRT_unpruned,
#                      n_fold = 3,
#                      seed=1,
#                      choice="min")
# GPRT=prune(GPRT_unpruned,cp=alpha/GPRT_unpruned$frame[1,"dev"])
#
# save(GPRT,file="GPRT.RData")
#
#
# tree=GPRT
#
# y_val_tree=matrix(c(1:dim(tree$frame)[1],
#                     as.numeric(row.names(tree$frame)),
#                     as.numeric(as.matrix(tree$frame)[,c("yval2.1","yval2.2")])),
#                   nrow=dim(tree$frame)[1],ncol=4,byrow=FALSE)
# colnames(y_val_tree)=c("where","node","sh","sc")
#
# number_of_terminal_nodes=sum(tree$frame[,"var"]=="<leaf>")
#
# where_test_data=as.vector(rpart.predict.leaves(tree, data_test, type = "where"))
#
# fit_sigma_cart=c()
# fit_gamma_cart=c()
#
# for(i in 1:dim(data_test)[1]){
#   fit_sigma_cart=c(fit_sigma_cart,y_val_tree[y_val_tree[,"where"]==where_test_data[i],"sc"])
#   fit_gamma_cart=c(fit_gamma_cart,y_val_tree[y_val_tree[,"where"]==where_test_data[i],"sh"])
# }
#
# par(mfrow=c(1,2))
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Sigma")
# lines(x,fit_sigma_cart,col="blue")
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Gamma")
# lines(x,fit_gamma_cart,col="blue")

learning_CART_GPD=function(data_sample){

  GPRT_unpruned=rpart(data = data_sample,
                      formula = Y ~ X,
                      method = GPRT_method_xi_in_R_plus,
                      #control = rpart.control(cp=0,minsplit=max(dim(data_sample)[1]/20,15), minbucket=max(dim(data_sample)[1]/60,7), maxcompete=0, maxsurrogate=0),
                      control = rpart.control(cp=0,minsplit=100, minbucket=50, maxcompete=0, maxsurrogate=0),
                      xval=0)

  #4 Pruning of the tree if it is not a root

  #a) By cross validation

  COMPUTE=FALSE
  if(length(GPRT_unpruned$splits)>0){
    if(nrow(GPRT_unpruned$splits)>0){
      COMPUTE=TRUE
    }
  }
  if(COMPUTE){
    cv=cross_val_gprt(data = data_sample,
                         formula = Y ~ X,
                         arbre = GPRT_unpruned,
                         n_fold = 3,
                         seed=1,
                         choice="min")
    alpha=cv[[1]]
    #GPRT=prune(GPRT_unpruned,cp=alpha/GPRT_unpruned$frame[1,"dev"])
    GPRT=prune_tree(rpart_object=GPRT_unpruned,CP=alpha)

    res_fit=GPRT
    fit=GPRT_unpruned
    ll_cv=list(cv[[2]],cv[[3]])
    res=list(res_fit,fit,ll_cv)

}else{
  GPRT=GPRT_unpruned

  res_fit=GPRT
  fit=GPRT_unpruned
  res=list(res_fit,fit,NULL)
}



  return(res)
}


pred_CART_GPD=function(tree,data_test){
  y_val_tree=matrix(c(1:dim(tree$frame)[1],
                      as.numeric(row.names(tree$frame)),
                      as.numeric(as.matrix(tree$frame)[,c("yval2.1","yval2.2")])),
                    nrow=dim(tree$frame)[1],ncol=4,byrow=FALSE)
  colnames(y_val_tree)=c("where","node","sh","sc")

  number_of_terminal_nodes=sum(tree$frame[,"var"]=="<leaf>")

  where_test_data=as.vector(rpart.predict.leaves(tree, data_test, type = "where"))

  fit_sigma_cart=c()
  fit_gamma_cart=c()

  for(i in 1:dim(data_test)[1]){
    fit_sigma_cart=c(fit_sigma_cart,y_val_tree[y_val_tree[,"where"]==where_test_data[i],"sc"])
    fit_gamma_cart=c(fit_gamma_cart,y_val_tree[y_val_tree[,"where"]==where_test_data[i],"sh"])
  }

  return(cbind(fit_sigma_cart, fit_gamma_cart))
}
