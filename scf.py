
from pyscf.lib import eig
from pyscf import gto , scf
import numpy as np
import scipy.linalg as sl

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 0.9184',
    basis = 'STO-3G',
)

def calculate_jk(P, ao_eri):
    n_ao = P.shape[0]
    J = np.zeros((n_ao, n_ao))
    K = np.zeros((n_ao, n_ao))
    for i in range(n_ao):
        for j in range(n_ao):
            for k in range(n_ao):
                for l in range(n_ao):
                    J[i,j] += ao_eri[i,j,k,l] * P[k,l] 
                    K[i,j] += ao_eri[i,k,j,l] * P[k,l]
    return J, K

def H_core(T, V, J, K):
    return T + V + J - 0.5 * K

def P_mat(C, occ):
    C_occ = C[:, 0:occ]
    return (C_occ @ C_occ.T) * 2

def eig_solver(S, H):
    eig_val , eig_vec = sl.eigh(H, S)
    return eig_val , eig_vec    

class SCF:
    def __init__(self, mol):
        self.mol = mol
        self.S = mol.intor('int1e_ovlp')
        self.T = mol.intor('int1e_kin')
        self.V = mol.intor('int1e_nuc')
        self.H_ao = self.T + self.V
        self.ao_eri = mol.intor('int2e')
        self.n_ao = len(self.mol.ao_labels())
        self.n_occ = (mol.nelectron // 2)
        self.C = np.diag(np.ones(self.n_ao))
        self.E_nuc = mol.energy_nuc()
        self.P_deq = []
        self.E_deq = [] 
        self.P, self.H = self.iguess()
        
    def isconverged(self):
        if len(self.P_deq) <= 1 or len(self.E_deq) <= 2:
            return False
        return np.allclose(self.P, self.P_deq[-1]) and abs(self.E_deq[-1] - self.E_deq[-2]) < 1e-16

    def iguess(self):
        P = P_mat(self.C, self.n_occ)
        J, K = calculate_jk(P, self.ao_eri)
        H = H_core(self.T, self.V, J, K)
        eig_val, eig_vec = eig_solver(self.S, H)
        self.C = eig_vec
        self.P_deq.append(P)
        self.E_deq.append(np.sum(eig_val))
        return P , H

    def iterator(self):
        not_converged = not self.isconverged()
        while not_converged:
            eig_val, eig_vec = eig_solver(self.S, self.H)
            self.C = eig_vec
            self.P = P_mat(self.C, self.n_occ)
            self.P_deq.append(self.P)
            J, K = calculate_jk(self.P, self.ao_eri)
            self.H = H_core(self.T, self.V, J, K)   
            E_elec = 0.5 * np.sum(self.P * (self.H + self.T + self.V)) + self.E_nuc
            not_converged = not self.isconverged()
            print("E = ", E_elec)
            print("not converged: ", not_converged)
            self.E_deq.append(E_elec)
            yield eig_val, eig_vec
            
    def runner(self):
        eles = []
        labels = self.mol.ao_labels()
        for label in labels:
            eles.append(label.split()[0])

        for eig_val, eig_vec in self.iterator():
            charges = {ele : 0 for ele in eles}
            for i in range(self.n_ao):
                for j in range(self.n_occ):
                    charges[eles[i]] += (float(eig_vec[i][j]) ** 2) * 2 - float(eles[i])
            print(charges)


if __name__ == "__main__":
    scf = SCF(mol)
    scf.runner()
