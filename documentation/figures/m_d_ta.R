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

par(mar=c(5,3.75,.75,.75))



beta <- 2000/7
s_1 <- 2000
s_2 <- 1750
tb <-  s_1+s_2
D <- ( -1200+tb):(1200+tb)
ta <- s_1 + s_2 + 250

dnormal <- dnorm(D,s_1+s_2,beta)
plot(D,dnormal, type="l",lwd=1,axes = F,ann = F)

axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1, at=0,las=0,cex.axis=1.33,line=-0.6)
axis(lwd=1,side=1, at=ta,labels=expression( t[a] ),las=0,cex.axis=1.33,line=-0.3)

mtext(text ="Density" ,side =2 ,line=2,cex=1.75)
mtext(text =expression( t[b] %~% N(sum(mu[i],i %in% A[b]),sum(beta^2+ sigma[i]^2,i %in% A[b])) ) ,side =1 ,line=4,cex=1.75)
abline(v=tb,lty=3)

#segments(0,0,0,dnorm(0,s_1+s_2,beta))
base <- rep(0,length(D))
xx <- c(D[D<=ta],rev(D[D<=ta]))
yy <- c(base[D<=ta],rev(dnormal[D<=ta]) )
polygon(xx,yy,col=rgb(0,0,0,0.4))

mid <- length(D)%/%2
h <- dnormal[mid]/4
text(D[mid],h, expression(t[b]<t[a]), col = 1, cex=1.75)

#######################################
# end 
dev.off()
setwd(oldwd)
par(oldpar, new=F)
#########################################