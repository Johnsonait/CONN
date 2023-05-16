from dataclasses import dataclass

import numpy as np

@dataclass
class Crystal():
    __m_vec = np.array([
        [0, -1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, -1, 1],
        [-1, 0, 1],
        [-1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [-1, 1, 0],
        [0, 1, 1],
        [-1, 0, 1],
        [1, 1, 0]
    ])/np.linalg.norm(np.array([0,1,1]))

    __s_vec = np.array([
        [-1, 1, 1],
        [-1, 1, 1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [-1, -1, 1],
        [-1, -1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, 1],
        [1, -1, 1]
    ])/np.linalg.norm(np.array([1,1,1]))

    n_slip = __s_vec.shape[0]

    def __init__(
                self,
                C11 = 200e9,
                C12 = 150e9,
                C44 = 75e9,
                Q = np.eye(3)
                ):

        self.C11 = C11
        self.C12 = C12
        self.C44 = C44

        self.Q = Q
        
        self.ELAS = self.__ELAS(C11=C11,C12=C12,C44=C44)

        self.SA = self.__SA()
        self.PA = self.__PA(self.SA)
        self.WA = self.__WA(self.SA)

        return

    def __SA():
        SA = np.zeros(Crystal.n_slip,3,3)
        for sys in range(Crystal.n_slip):
            m_vec = Crystal.__m_vec[sys,:]
            s_vec = Crystal.__s_vec[sys,:]

            SA[sys,:,:] = np.outer(s_vec,m_vec)
        return

    def __PA(self,SA):
        PA = np.zeros(Crystal.n_slip,3,3)
        for sys in range(Crystal.n_slip):
            SA_sys = SA[sys,:,:]
            PA[sys,:,:] = 0.5*(SA_sys + np.transpose(SA_sys))
        return PA

    def __WA(self,SA):
        WA = np.zeros(Crystal.n_slip,3,3)
        for sys in range(Crystal.n_slip):
            SA_sys = SA[sys,:,:]
            WA[sys,:,:] = 0.5*(SA_sys - np.transpose(SA_sys))
        return WA


    def __ELAS(self,C11,C12,C44):
        ELAS = np.array([
            [C11,C12,C12,0,  0,  0  ],
            [C12,C11,C12,0,  0,  0  ],
            [C12,C12,C11,0,  0,  0  ],
            [0,  0,  0,  C44,0,  0  ],
            [0,  0,  0,  0,  C44,0  ],
            [0,  0,  0,  0,  0,  C44]
        ])
        return