# @title Simplex method

import numpy as np
import random

class Simplex:
  def __init__(self, condition, objective, ranks_of_A, ranks_of_b):
    
    self.condition, self.objective, self.ranks_of_A, self.ranks_of_b = condition, objective, ranks_of_A, ranks_of_b  
    #↑条件関数＆目的関数などの代入
    self.row_cond      = self.condition.shape[0]  #行数  
    self.list_cond     = self.condition.shape[1]  #列数
    self.row_A         = self.ranks_of_A.shape[0] #Aの行数
    self.list_A        = self.ranks_of_A.shape[1] #Aの列数
    self.basic_var_ind = []                 #基底変数のインデックス
    self.basic_var     = np.empty((self.row_cond,self.list_cond))#基底変数行列
    self.non_basic_ind = []                #非基底変数のインデックス（今はまだ空）
    self.non_basic     = np.empty((self.row_cond, self.list_cond),dtype=int)#非基底変数行列
    
#線形独立なベクトルを選んでくる関数
  def determin_basic_var(self):
    self.basic_var         = self.ranks_of_A[:, 0:1]  #基底変数に条件関数の1列目を付け加えている(後ろのスライスをミスると、1次元になってしまいエラーをはくので注意)
    new_judgement_ranks    = self.basic_var


    for i in range(1,self.list_A):
      new_judgement_ranks  = np.hstack((self.basic_var,self.ranks_of_A[:,i:i+1]))   #列を追加して判定用行列に付け加えている


      #判定用の行列のrankに新たな列を付け加えたもののrankが、その行列全体の列数と同じになる限り付け加え続ける
      #このループが終わった後には、線型独立のものしか残らない（線型従属なものを加えてしまった場合、rank=列数になることはありえないため）
      if np.linalg.matrix_rank(new_judgement_ranks)== new_judgement_ranks.shape[1]:      
        self.basic_var_ind.append(i)
        #judgement_ranks  = np.hstack((judgement_ranks,self.condition[i]))
        self.basic_var   = np.hstack((self.basic_var,self.ranks_of_A[:,i:i+1]))

      else:
      
        if len(self.non_basic_ind)==0:
          self.non_basic_ind.append(i)
          self.non_basic = self.ranks_of_A[:,i:i+1]

        else:
          self.non_basic = np.hstack((self.non_basic,self.ranks_of_A[:,i:i+1]))
          self.non_basic_ind.append(i)

    return self.basic_var


  def optimizeable(self):   #最適化の余地があるかを判定。すでに最適化されていたらFalseを返す
    if np.amin(self.objective) > 0:
      return False
    else:
      return True

  def next_basic_vars(self):    #目的関数の中で最小の列の添字を返す。これが次の基底変数となる。
    return np.argmin(self.objective)


  def next_nonbasic_vars(self):  #b/Aが最小の行の添字を返す。これとnext_basic_varsを入れ替えてあげる。
    
    K                     = self.next_basic_vars()
    self.newranks         = self.ranks_of_b / self.ranks_of_A[:, K:K+1]
    self.next_nonbasic_vars  = np.argmin(self.newranks,  axis = 1)
    return self.next_nonbasic_vars



  def reduce_row(self):    #非基底変数と基底変数の入れ替え＆掃き出し

    next_nonbasic_vars = self.next_nonbasic_vars()
    next_basic_vars    = self.next_basic_vars()

    #掃き出しはここから
    self.condition[Simplex.next_nonbasic_vars] /= self.ranks_of_A[next_nonbasic_vars,next_basic_vars]
    self.objective                             /= self. ranks_of_A[next_nonbasic_vars,next_basic_vars]

    return self.objective
    #self.ranks_of_A -= 






if __name__ == "__main__":
    
  condition  = np.array([[5,2,1,0,30],[1,2,0,1,14]]) #条件関数
  objective  = np.array([-5,-4,0,0,0])  #目的関数
  ranks_of_A = np.array([[5,2,1,0],[1,2,0,1]])#条件関数の右辺
  ranks_of_b = np.array([[30],[14]])          #条件関数の左辺だけ抽出したもの

  simplex = Simplex(condition, objective, ranks_of_A, ranks_of_b)
  
  # while(simplex.optimizeable()):
  
  #print(simplex.is_replecable())
  print(simplex.determin_basic_var())
  #print(simplex.next_nonbasic_vars())
 # print(simplex.optimizeable())
  print(simplex.reduce_row())
