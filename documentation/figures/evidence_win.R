oldpar <- par(no.readonly = TRUE)
oldwd <- getwd()
this.dir <- dirname(parent.frame(2)$ofile)
nombre.R <-  sys.frame(1)$ofile
require(tools)
nombre <- print(file_path_sans_ext(nombre.R))
pdf(paste0(nombre,".pdf"), width = 8, height = 5  )
setwd(this.dir)
###############################

par(mar=c(3.75,3.75,0.25,0.25))


beta <- 25/6
#caso Equiop muy superior
mu <- c(25,30,25,25)
sigma <- c(8,1,1,1)
mu1_grilla <- seq(mu[1]-25,mu[1]+25,by=0.1)
posterior_ganador <- function(s_1_f,s_1_p,mu= c(25,20,20,20),sigma=c(8,1,1,1),beta=25/6){
  return(dnorm(s_1_f,mu[1],sigma[1])*pnorm(s_1_p,sum(mu[3:4])-(sum(mu[1:2])-mu[1]),sqrt(sum(sigma^2+beta^2)-sigma[1]^2)))
  
}
posterior_ganador2 <- function(s_1_f,s_1_p,mu= c(25,20,20,20),sigma=c(8,1,1,1),beta=25/6){
  return(dnorm(s_1_f,mu[1],sigma[1])*pnorm(0,sum(mu[3:4])-(sum(mu[1:2])-mu[1]+s_1_p),sqrt(sum(sigma^2+beta^2)-sigma[1]^2)))
}
posterior_ganador2 <- function(s_1_f,s_1_p,mu= c(25,20,20,20),sigma=c(8,1,1,1),beta=25/6){
  delta_s1 <- sum(mu[1:2])-sum(mu[3:4])-mu[1]+s_1_p
  vartheta1 <-sqrt(sum(sigma^2+beta^2)-sigma[1]^2)
  return(dnorm(s_1_f,mu[1],sigma[1])*pnorm(delta_s1/vartheta1))
}

m <- outer(mu1_grilla,mu1_grilla,posterior_ganador)
m2 <- outer(mu1_grilla,mu1_grilla,posterior_ganador2)
#m[250,250] ;m2[250,250]

levels <- seq(min(m),max(m),length.out = 11)

#abline(v=max_post,lty=2)
#axis(lwd=0,side=1, at=max_post,labels="max" ,las=0,cex.axis=1,line=-0.9)


beta <- 25/6

posterior_ganador <- function(s_1,mu,sigma,beta=25/6){
  return(dnorm(s_1,mu[1],sigma[1])*pnorm(s_1,sum(mu[3:4])-(sum(mu[1:2])-mu[1]),sqrt(sum(sigma^2+beta^2)-sigma[1]^2)))
}
prior <- function(s_1,mu,sigma,beta=25/6){
  return(dnorm(s_1,mu[1],sigma[1]))
}
sorpresa_de_ganar <- function(s_1,mu,sigma,beta=25/6){
  return(pnorm(s_1,mu[1]-(sum(mu[1:2])-sum(mu[3:4])),sqrt(sum(sigma^2+beta^2)-sigma[1]^2)))
}

posterior <- apply(diag(dim(m)[1],dim(m)[1])*m,2,sum)
index_max <- which.max(posterior)
max_post <- mu1_grilla[index_max]

ejes <- function(){
  axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
  axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
  axis(lwd=0,side=1, at=0, labels=0,cex.axis=1.75,line=-0.3)
  mtext(text= expression(Skill[a]),side =1,line=2,cex=2.5)
  mtext(text ="Density" ,side =2,line=1,cex=2.5)
  
}

x.min <- min(mu1_grilla)
x.max <- max(mu1_grilla)
posterior2 <- sorpresa_de_ganar(mu1_grilla,mu,sigma)*prior(mu1_grilla,mu,sigma)/max(prior(mu1_grilla,mu,sigma))


plot(mu1_grilla, prior(mu1_grilla,mu,sigma)/max(prior(mu1_grilla,mu,sigma))
     ,lty=2,lwd=3,type="l",axes = F,ann = F,xlim=c(x.min,x.max+2),col="white")


yy <- c(posterior2,rep(0,length(mu1_grilla)))
xx <- c(mu1_grilla,rev(mu1_grilla))      
white=0.9
polygon(xx,yy,col=rgb(1*white,0.6*white,0,0.8),border=F)

lines(mu1_grilla, prior(mu1_grilla,mu,sigma)/max(prior(mu1_grilla,mu,sigma)),lty=2,lwd=3)
lines(mu1_grilla, sorpresa_de_ganar(mu1_grilla,mu,sigma),lty=1,lwd=3)
ejes()
mid = mu[1]-(sum(mu[1:2])-sum(mu[3:4]))

yy <- c(sorpresa_de_ganar(mu1_grilla,mu,sigma),rep(1,length(mu1_grilla)))
xx <- c(mu1_grilla,rev(mu1_grilla))      
polygon(xx,yy,col=rgb(0,0,0,0.075),border=F)


lines(mu1_grilla,posterior2 ,lty=3,lwd=3)

legend(mu1_grilla[13*length(mu1_grilla)%/%20],0.95,lty = c(2,1,3),lwd=c(2,2,2),
       legend = c("Prior","Likelihood",expression("Posterior"%prop%"") ),bty = "n",cex = 2)


points(mu[1],sorpresa_de_ganar(mu[1],mu,sigma),pch=19,cex=1.5)
points(max_post,posterior2[index_max],cex=1.5)


text(max_post,0.15,"Evidence",srt=0, cex=2.2)


#######################################
# end 
dev.off()
system(paste("pdfcrop -m '0 0 0 0'",paste0(nombre,".pdf") ,paste0(nombre,".pdf")))
setwd(oldwd)
par(oldpar, new=F)
#########################################
