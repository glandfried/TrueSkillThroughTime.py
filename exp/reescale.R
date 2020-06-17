"
Observaci\'on:
    Un diferencia de habilidad de una escala representa 0.76 de ganar
    Si tenemos incertidumbre respecto de la habilidad, si bien la probabilidad de ganar es la misma,
    nuestra predicci\'on cambia, reduciendo la probabilidad real de ganar.  
"
probabilidad_de_ganar <- function(n=100000,escala=1,mu_a=30,distancia_b=NA,incertidumbre=0){
  if (is.na(distancia_b)){ distancia_b = escala  }
  mu_b = mu_a - distancia_b 
  ana <- rnorm(n,mu_a, sqrt(escala^2 + incertidumbre^2) )
  berta  <- rnorm(n,mu_b, sqrt(escala^2 + incertidumbre^2) )
  return(sum(ana > berta)/n)
}
probabilidad_de_ganar()

"
Problema (Escala Handicap):
    Cada piedra de handicap en Go agrega $x$ habilidad a las personas.
    Una primera aproximaci\'on encontramos que $x ~ 0.2 + (h-1) * 0.83$
    Este estiimaci\'on se ver\'a corregida cuando agregemos las estimaciones
    de jugar segundo y de komi. Para simplificar, supongamos que la habilidad 
    que agrega cada piedra es de $0.85$ bajo una escala por defecto de $25/6$.
    Nosotres queremos determinar una escala $\beta$ que lleve la habilidad que
    agregan las piedras a 1. Lo que se debe mantener constante en la probabilidad
    de ganar.
    1) Primero miramos la probabilidad de ganar modificando al habilidad de a $0.85$
    2) Luego elegimos una escala alternativa que haga que un $1.0$ de habilidad
       modifique la probabilidad de ganar de la misma forma que en (1)
    3) Verifico que sea independiente a la incertidumbre. 
"
# 1) Probabilidad de ganar modificando al habilidad de a $0.85$
handicap <- c()
for(h in seq(2,9)){
  handicap <- c(handicap, probabilidad_de_ganar(n=10^6,escala=25/6,distancia_b=(h-1)*0.85))
}
plot(seq(2,9),handicap)
# 2) Escala alternativa que haga que un $1.0$ de habilidad tenga mismo efecto sobre probabilidad de ganar
nueva_escala = (25/6)/0.85
nuevo_handicap <- c()
for(h in seq(2,9)){
  nuevo_handicap <- c(nuevo_handicap, probabilidad_de_ganar(n=10^6,escala=nueva_escala,distancia_b=h-1))
}
points(seq(2,9),nuevo_handicap)
# 3) Verifico que la dependencia respecto de la incertidumbre
handicap <- c()
for(h in seq(2,9)){
  handicap <- c(handicap, probabilidad_de_ganar(n=10^6,escala=25/6,distancia_b=(h-1)*0.85,incertidumbre=25/6))
}
plot(seq(2,9),handicap)
nueva_escala = (25/6)/0.85
nuevo_handicap <- c()
for(h in seq(2,9)){
  nuevo_handicap <- c(nuevo_handicap, probabilidad_de_ganar(n=10^6,escala=nueva_escala,distancia_b=h-1,incertidumbre=nueva_escala))
}
points(seq(2,9),nuevo_handicap)
