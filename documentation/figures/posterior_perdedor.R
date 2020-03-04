###########################################
# Header
oldpar <- par(no.readonly = TRUE)
oldwd <- getwd()
this.dir <- dirname(parent.frame(2)$ofile)
nombre.R <-  sys.frame(1)$ofile
require(tools)
nombre <- print(file_path_sans_ext(nombre.R))
pdf(paste0(nombre,".pdf"))
setwd(this.dir)
#setwd("~/gaming/materias/inferencia_bayesiana/trabajoFinal/imagenes")
#####################################

par(mar=c(3.75,3.75,0.25,0.25))

beta <- 25/6
#caso Equiop muy superior
mu <- c(25,30,25,25)
sigma <- c(8,1,1,1)
mu1_grilla <- seq(mu[1]-25,mu[1]+25,by=0.1)
posterior_ganador <- function(s_1_f,s_1_p,mu= c(25,20,20,20),sigma=c(8,1,1,1),beta=25/6){
  return(dnorm(s_1_f,mu[1],sigma[1])*(1-pnorm(s_1_p,sum(mu[3:4])-(sum(mu[1:2])-mu[1]),sqrt(sum(sigma^2+beta^2)-sigma[1]^2))))
  
}

m <- outer(mu1_grilla,mu1_grilla,posterior_ganador)
levels <- seq(min(m),max(m),length.out = 11)
image(mu1_grilla,mu1_grilla,m,col=rev(gray.colors(10,start=0.2,end=0.95)),breaks = levels,useRaster=T,
      ylab="",xlab="",axes=F)
contour(mu1_grilla,mu1_grilla,m,drawlabels=F,levels = levels,add = T,col=rev(gray.colors(11,start=0,end=0.6)),lwd=1.1)
mtext(text=expression(N(s[1]~";"~mu[1],sigma[1]^2 )) ,side =1,line=2.75,cex=1.75)
mtext(text =expression(Phi(-delta(s[1])/vartheta[1])) ,side =2,line=1.5,cex=1.75)
abline(v=mu[1],lty=3)
abline(h=mu[1],lty=3)
abline(c(0,1))
#lines(c(0,1)*(s_2+s_2-500),c(1,0)*(s_2+s_2-500))
axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1, at=mu[1], labels=expression(mu[1]),cex.axis=1.25,line=-0.85,tck=0.015)
axis(lwd=0,side=2, at=mu[1], labels=expression(delta),cex.axis=1.25,line=-1,tck=0.015)
posterior <- apply(diag(dim(m)[1],dim(m)[1])*m,2,sum)
index_max <- which.max(posterior)
max_post <- mu1_grilla[index_max]
points(max_post,max_post,cex=1.5)
points(mu[1],mu[1],pch=19,cex=1.5)


beta <- 25/6

posterior_ganador <- function(s_1,mu,sigma,beta=25/6){
  return(dnorm(s_1,mu[1],sigma[1])*(1-pnorm(s_1,sum(mu[3:4])-(sum(mu[1:2])-mu[1]),sqrt(sum(sigma^2+beta^2)-sigma[1]^2))))
}
prior <- function(s_1,mu,sigma,beta=25/6){
  return(dnorm(s_1,mu[1],sigma[1]))
}
sorpresa_de_ganar <- function(s_1,mu,sigma,beta=25/6){
  return(1-pnorm(s_1,mu[1]-(sum(mu[1:2])-sum(mu[3:4])),sqrt(sum(sigma^2+beta^2)-sigma[1]^2)))
}

plot(mu1_grilla, sorpresa_de_ganar(mu1_grilla,mu,sigma),type="l",axes = F,ann = F,lty=1)
lines(mu1_grilla, prior(mu1_grilla,mu,sigma)/max(prior(mu1_grilla,mu,sigma)),lty=2)
posterior2 <- sorpresa_de_ganar(mu1_grilla,mu,sigma)*prior(mu1_grilla,mu,sigma)/max(prior(mu1_grilla,mu,sigma))
lines(mu1_grilla,posterior2 ,lty=1,lwd=2)

axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1, at=0, labels=0,cex.axis=1.25,line=-0.3)
axis(lwd=0,side=1, at=mu[1], labels=expression(mu[1]),cex.axis=1.25,line=-0.85,tck=0.015)
points(mu[1],sorpresa_de_ganar(mu[1],mu,sigma),pch=19,cex=1.5)
points(max_post,posterior2[index_max],cex=1.5)

mid = mu[1]-(sum(mu[1:2])-sum(mu[3:4]))
abline(v=mu[1],lty=3)
abline(h=sorpresa_de_ganar(mu[1],mu,sigma),lty=3)

#y <- 0.1
#segments(mid,y,mu[1],y,lwd=2)
#segments(mid,y+.01,mid,y-0.01,lwd=2)
#segments(mu[1],y+.01,mu[1],y-.01,lwd=2)
#text(mid + (mu[1]-mid)/2, y+0.02,expression(delta) ,cex=1.33)

yy <- c(sorpresa_de_ganar(mu1_grilla,mu,sigma),rep(1,length(mu1_grilla)))
xx <- c(mu1_grilla,rev(mu1_grilla))      
polygon(xx,yy,col=rgb(0,0,0,0.3),border=F)

text(mu1_grilla[4*length(mu1_grilla)%/%5],0.6, "Surprise",srt=0, cex=1.75)

mtext(text= expression(s[1]),side =1,line=2,cex=1.75)
mtext(text ="Density" ,side =2,line=1,cex=1.75)

legend(mu1_grilla[1]-3,0.74,lty = c(1,2,1),lwd=c(1,1,2),
       legend = c("Likelihood","Prior","Posterior"),bty = "n",cex = 1.5)


#######################################
# end 
dev.off()
setwd(oldwd)
par(oldpar, new=F)
#########################################