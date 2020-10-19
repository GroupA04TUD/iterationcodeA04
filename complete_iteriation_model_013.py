from math import *

s = 100                                             #avarage surface area reference aircraft
C_l_landing = 2.8
C_l_take_of = 1.6
C_l_max_clean = 1.4
A = 9.8
run_iteration_001 = True
iteration_check = True
iteration_count = 0.0

while run_iteration_001 == True:
    s_check = s
    
    if iteration_check == True:
        run_iteration_002 = True
        
    if iteration_check == False:
        run_iteration_002 = False
        #class II weight estimation

        I_n_c = 1.0/0.0254
        F_t_c = 1.0/0.3048
        F_t_c_2 = 1.0/(0.3048**2)
        F_t_c_3 = 1.0/(0.3048**3)
        L_b_c = 1.0/0.45359237
        K_T_s_c = 1.0/0.514444
        Gallon_c = 1.0/3.78541178
        Wmto_lb = Wmto*L_b_c
        noice = 2.1 + ((24000.0)/(Wmto_lb+10000.0))
        N_z = 1.5*noice
        
        
        W_dg = L_b_c*(Wmto/9.80665)                 #design weight
        S_w = F_t_c_2*s                             #wing area
        B_w = F_t_c*b                               #wing span
        t_c_root = 0.12                             #thickness to cord ratio at the root
        S_csw = F_t_c_2*((S_ail*2)+(s_flap*2))      #controlsurface area wing mounted
        F_w = F_t_c*1.0                             #fuselage width at horizontal tail intersection
        B_h = F_t_c*bH                              #horizontal tail span
        S_ht = F_t_c_2*S_h                          #horizontal tail surface area
        L_t_h = F_t_c*5.18                          #tail lenght wing quarter mac till horizontal tail quarter mac
        L_t_v =F_t_c*5.18                           #tail lenght wing quarter mac till vertical tail quarter mac
        L_t = (L_t_h +L_t_v)/2.0                    #average ^
        Lab1_4_htail = Lab1_4                       #quarter cord sweep angle of the horizontal tail
        A_htail = AH                                #aspect ratio of the horizontal tail
        S_e = F_t_c_2*4.48                          #elevator surface area
        H_t_H_v = 0.0                               #1.0 for T-tail configuration
        S_vt = F_t_c_2*S_v                          #surface area vertical tail
        Lab1_4_vtail = Lab1_4_v                     #quarter cord sweep angle vertical tail
        A_vtail = AV                                #aspect ration vertical tail
        t_c_root_vtail = 0.15                       #thickness to cord ratio at the root of the vertical tail
        L_fuselage = F_t_c*l_f                      #fuselage lenght excludes radome and tail cap
        D_fuselage = F_t_c*(d_f_outer-d_f_inner)    #fuselage structural depth
        W_l = L_b_c*((We+Wfres)/9.80665)            #landing design weight
        N_gear = 3.0                                #number of gear
        N_l = N_gear*1.5                            #ultimate landing load factor
        L_m = I_n_c*1.0                             #lenght of main landing gear
        L_n =I_n_c*1.0                              #lenght of nose landing gear
        N_mw = 4.0                                  #number of main wheels
        N_nw = 2.0                                  #number of nose wheels
        N_mss = 2.0                                 #number of main gear shock struts
        V_stall = K_T_s_c*v_stall_landing           #stall speed
        N_lt = F_t_c*4.81                           #nacelle lenght
        N_w = F_t_c*1.32                            #nacelle width
        W_en = L_b_c*(17828.4897/9.80665)           #weight of an engine
        W_ec = 2.331*(W_en**0.901)                  #weight of engine and contents
        N_en = 2.0                                  #number of engines
        S_n = F_t_c_2*2*pi*N_lt*(0.5*N_w)           #nacelle wetted area
        L_ec = F_t_c*(14.38*N_en)                   #length form engine front till cockpit total for multiple engines
        V_t = Gallon_c*((Wf/9.80665)/775)           #total fuel volume
        V_i = Gallon_c*V_T                          #total integral tank volume
        V_p = Gallon_c*0                            #self healing protected tanks volume
        N_t = 2.0                                   #number of fuel tanks
        N_f = 5.0                                   #number of functions preformed by controls (4-7)
        N_m = 1.0                                   #number of mechanical functions
        S_cs = F_t_c_2*((S_ail*2)+(s_flap*2)+4.48)  #total area of control surfaces
        I_y = 39598140.552692                       #yawing moment of inertia
        N_c = 3.0                                   #number of crew
        L_f = F_t_c*l_f                             #total fuselage lenght
        R_kva = 50.0                                #system electrical rating
        L_a = F_t_c*13.72                           #electrical routing distance form generators till cockpit
        N_gen = 2.0                                 #number of generators typically = N_en
        W_uav = 1100.0                              #uninstalled avionics weight typically 800-1400
        W_c = L_b_c*2563.0                          #maximun cargo weight
        S_f = F_t_c_2*111.44                        #fuselage wetted area
        N_p = 21.0                                  #number of onboard personnel(crew plus passengers)
        V_pr = F_t_c_3*300.0                        #pressurized volume

        K_uht = 1.0                                 #1.43 for a moving tail
        K_y = 0.3*L_t                               #pitching radius of gyration
        K_z = L_t                                   #yawing radius of gyration
        K_door = 1.0                                #if plane has no cargo doors 1.0
        K_L_g = 1.2                                 #1.2 for fuselage mounted landing gear
        K_ws = 0.75*((1+2*lab)/(1+lab))*((B_w*tan(radians(Lab1_4)))/L_fuselage)
        K_mp = 1.0                                  #1.26 for kneeling gear
        K_np = 1.0                                  #1.15 for kneeling gear
        K_ng = 1.0                                  #1.017 for pylon mounted nacelle
        K_r = 1.0                                   #1.133 for reciprocating engine
        K_tp = 1.0                                  #0.793 for a trupo prop

        W_wing = 0.0051*((W_dg*N_z)**0.557)*(S_w**0.649)*(A**0.5)*(t_c_root**-0.4)*((1+lab)**0.1)*((cos(radians(Lab1_4)))**-1)*(S_csw**0.1)
        W_horizontal_tail = 0.0379*(K_uht)*((1+(F_w/B_h))**-0.25)*(W_dg**0.639)*(N_z**0.1)*(S_ht**0.75)*(L_t_h**-1)*(K_y**0.704)*((cos(radians(Lab1_4_htail)))**-1)*(A_htail**0.166)*((1+(S_e/S_ht))**0.1)
        W_vertical_tail = 0.0026*((1+(H_t_H_v))**0.225)*(W_dg**0.556)*(N_z**0.536)*(L_t_v**-0.5)*(S_vt**0.5)*(K_z**0.875)*((cos(radians(Lab1_4_vtail)))**-1)*(A_vtail**0.35)*(t_c_root_vtail**-0.5)

        W_fuselage = 0.3280*(K_door)*(K_L_g)*((W_dg*N_z)**0.5)*(L_fuselage**0.25)*(S_f**0.302)*((1+K_ws)**0.04)*((L_fuselage/D_fuselage)**0.1)

        W_main_landing_gear = 0.0106*(K_mp)*(W_l**0.888)*(N_l**0.25)*(L_m**0.4)*(N_mw**0.321)*(N_mss**-0.5)*(V_stall**0.1)
        W_nose_landing_gear = 0.032*(K_np)*(W_l**0.646)*(N_l**0.2)*(L_n**0.5)*(N_nw**0.45)

        w_nacelle_group = 0.6724*(K_ng)*(N_lt**0.10)*(N_w**0.294)*(N_z**0.119)*(W_ec**0.611)*(N_en**0.984)*(S_n**0.224)

        W_engine_controls = 5.0*N_en + 0.80*L_ec

        W_starter_pneumatic = 49.19*(((N_en*W_en)/(1000))**0.541)

        W_fuel_system = 2.405*(V_t**0.606)*((1+(V_i/V_t))**-1.0)*(1+(V_p/V_t))*(N_t**0.5)

        W_flight_controls = 145.9*(N_f**0.554)*((1+(N_m/N_f))**-1.0)*(S_cs**0.20)*((I_y*10**-6)**0.07)

        W_intruments = 4.509*K_r*K_tp*(N_c**0.541)*N_en*((L_f+B_w)**0.5)

        W_hydraulics = 0.2673*N_f*((L_f+B_w)**0.5)

        W_electrical = 7.291*(R_kva**0.782)*(L_a**0.346)*(N_gen**0.10)

        W_avionics = 1.73*(W_uav**0.983)

        W_furnishings = 0.0577*(N_c**0.1)*(W_c**0.393)*(S_f**0.75)

        W_air_conditioning = 62.36*(N_p**0.25)*((V_pr/1000)**0.604)*(W_uav**0.10)

        W_anti_ice = 0.002*W_dg

        W_handling_gear = (3.0*10**-4)*W_dg

        W_e_lb = W_wing + W_horizontal_tail + W_vertical_tail + W_fuselage + W_main_landing_gear + W_nose_landing_gear + w_nacelle_group + W_engine_controls + W_starter_pneumatic + W_fuel_system + W_flight_controls + W_intruments + W_hydraulics + W_electrical + W_avionics + W_furnishings + W_air_conditioning + W_anti_ice + W_handling_gear
        W_e_N = (W_e_lb*0.45359237)*9.80665
        
        We = W_e_N
        Wmto = (We-xy)/(a)
        Wf = (1-mff)* Wmto
        
        # class 2 drag estimations
        #input parameters
        M_cruise = 0.85
        f_turbelent_fus = 0.95
        f_laminar_fus = 0.05
        f_turbelent_wing_tail = 0.9
        f_laminar_wing_tail = 0.1
        k_paint = 0.634*(10**(-5))
        mac_k = C_mac/k_paint
        l_fuselage = 18.36
        l_nosecone = 5.8
        l_center = 5.31
        l_tailcone = 7.25
        l_nacelle = 4.81
        d_nacelle = 1.32
        d_fuselage = 2.9
        lambda_m_wing = 0.667
        x_c_max_wing = 0.37
        t_c_wing = 0.12
        lambda_m_VT = 0.75
        x_c_max_VT = 0.3
        t_c_VT = 0.12
        lambda_m_HT = 0.65
        x_c_max_HT = 0.3
        t_c_HT = 0.08

        # RE calculations
        #Re inputs
        cutoff_Re_wing = (M_cruise**(1.16))*44.62*(mac_k**(1.053))
        cutoff_Re_fuselage = 44.62*((l_fuselage/k_paint)**1.053)*(M_cruise**1.16)
        cutoff_Re_nacelle = 44.62*((l_nacelle/k_paint)**1.053)*(M_cruise**1.16)
        cutoff_Re_horizontaltail = 44.62*((MACH/k_paint)**1.053)*(M_cruise**1.16)
        cutoff_Re_verticaltail = 44.62*((MACV/k_paint)**1.053)*(M_cruise**1.16)
        Re_cruise_fuselage = ((Re_cruise*l_fuselage)/C_mac)
        Re_cruise_nacelle = (Re_cruise*l_nacelle)/C_mac
        Re_cruise_horizontaltail = (Re_cruise*MACH)/C_mac
        Re_cruise_verticaltail = (Re_cruise*MACV)/C_mac
        #final Re values
        Re_fuselage = min(Re_cruise_fuselage,cutoff_Re_fuselage)
        Re_wing = min(Re_cruise,cutoff_Re_wing)
        Re_nacelle = min(Re_cruise_nacelle,cutoff_Re_nacelle)
        Re_verticaltail = min(Re_cruise_verticaltail,cutoff_Re_verticaltail)
        Re_horizontaltail = min(Re_cruise_horizontaltail,cutoff_Re_horizontaltail)

        # wing drag
        C_f_turbelent_wing = 0.455/((log10(Re_wing)**(2.58))*(1+(0.144*M_cruise**2)**0.65))
        C_f_laminar_wing = 1.328/(sqrt(Re_wing))
        FF_wing = (1+((0.6/x_c_max_wing)*t_c_wing))+(100*(t_c_wing**4)*(1.34*(M_cruise**0.18)*(cos(lambda_m_wing)**0.28)))

        #fuselage drag
        C_f_turbulent_fuselage = 0.455/(((log10(Re_fuselage))**2.58)*((1+(0.144*(M_cruise**2)))**0.65))
        C_f_laminar_fuselage = 1.328/(sqrt(Re_fuselage))
        upsweep_fuselage = (10*pi)/180
        A_max_fuselage = pi*(d_fuselage/2)**2
        f_fuselage = l_fuselage/ d_fuselage
        FF_fuselage = 1+(60/(f_fuselage**3))+(f_fuselage/400)

        #vertical tail drag
        C_f_turbulent_verticaltail = 0.455/(((log10(Re_verticaltail))**2.58)*((1+(0.144*(M_cruise**2)))**0.65))
        C_f_laminar_verticaltail = 1.328/sqrt(Re_verticaltail)
        FF_VT = (1+((0.6/x_c_max_VT)*t_c_VT))+(100*(t_c_VT**4)*(1.34*(M_cruise**0.18)*(cos(lambda_m_VT)**0.28)))

        #horizontal tail drag
        C_f_turbulent_horizontaltail = 0.455/(((log10(Re_horizontaltail))**2.58)*((1+(0.144*(M_cruise**2)))**0.65))
        C_f_laminar_horizontaltail = 1.328/sqrt(Re_horizontaltail)
        FF_HT = (1+((0.6/x_c_max_HT)*t_c_HT))+(100*(t_c_HT**4)*(1.34*(M_cruise**0.18)*(cos(lambda_m_HT)**0.28)))

        #engine nacelle drag
        C_f_turbulent_nacelle =0.455/(((log10(Re_nacelle))**2.58)*((1+(0.144*(M_cruise**2)))**0.65))
        C_f_laminar_nacelle = 1.328/(sqrt(Re_nacelle))
        f_nacelle =  l_nacelle/d_nacelle                           
        FF_nacelle = 1+0.35/f_nacelle
                            
        #HLD drag
        defec_angle_landing = 40
        c_flaps_c = 0.35
        SrefTE = Sref_trailing
        SrefLE = Sref_leading
        Fflap = 0.0074

        #wetted areas
        S_wetted_wing = s*2*1.07
        S_wetted_HT = S_h*2*1.05
        S_wetted_VT = S_v*2*1.05
        S_wetted_nacelle = 4*pi*(d_nacelle/2)*l_nacelle
        S_wetted_fuselage_1 = pi*d_fuselage
        S_wetted_fuselage_2 = ((4*(l_nosecone**2)+(d_fuselage**2)/4)**1.5)-((d_fuselage**3)/8)
        S_wetted_fuselage_3 = S_wetted_fuselage_2/(3*(l_nosecone**2))
        S_wetted_fuselage_4 = (-1*d_fuselage)+(4*l_center)+(2*sqrt((l_tailcone**2)+((d_fuselage**2)/4)))
        S_wetted_fuselage_total = (S_wetted_fuselage_1)*((S_wetted_fuselage_3+S_wetted_fuselage_4)/4)
        
        #flat plate coeff
        flat_pl_wing = (C_f_turbelent_wing*f_turbelent_wing_tail) + (C_f_laminar_wing*f_laminar_wing_tail)
        flat_pl_fuselage = (f_turbelent_fus*C_f_turbulent_fuselage) + (C_f_laminar_fuselage*f_laminar_fus)
        flat_pl_tail = (f_turbelent_wing_tail*(C_f_turbulent_verticaltail+C_f_turbulent_horizontaltail))+(f_laminar_wing_tail*(C_f_laminar_verticaltail+C_f_laminar_horizontaltail))
        flat_pl_engine = (C_f_turbulent_nacelle*f_turbelent_fus) + (C_f_laminar_nacelle*f_laminar_fus)

        # form factor coeff
        form_fact_wing = FF_wing 
        form_fact_fuselage = FF_fuselage
        form_fact_tail = FF_HT + FF_VT
        form_fact_engine = FF_nacelle
    
        #interference drag
        inter_wing = 1.25
        inter_fuselage = 1
        inter_tail = 1.04
        inter_engine = 1.3

        # misc drag
        wave_drag = 0
        misc_fuselage = 0
        misc_upsweep = (3.83*((upsweep_fuselage)**2.5)*A_max_fuselage)/s
        misc_leakage = 0.035
        misc_flaps = Fflap*c_flaps_c *((SrefTE+SrefLE)/s)*(defec_angle_landing-10)

        # total drag
        cd0_wing = form_fact_wing*inter_wing*flat_pl_wing*S_wetted_wing 
        cd0_fuselage = form_fact_fuselage*inter_wing*flat_pl_fuselage*S_wetted_fuselage_total 
        cd0_tail= form_fact_tail*inter_tail*flat_pl_tail*(S_wetted_VT+S_wetted_HT) 
        cd0_engine = form_fact_engine*inter_engine*flat_pl_engine*S_wetted_nacelle
        cd0_total = (1/s)*(cd0_wing + cd0_fuselage + cd0_tail + cd0_engine) + (wave_drag + misc_fuselage + misc_upsweep)

        total_cd0 = misc_leakage*cd0_total + cd0_total
        Cd0 = total_cd0

    while run_iteration_002 == True:
        #class I weight estimation
        #input
        e = 0.95
        Cj_c = 0.631/(3600*9.80665)
        Cj_l =0.4/(3600*9.80665)
        V = 250.8
        R = 10500000
        g = 9.80665
        Cd0 = 0.003*4.5
        E = 2700
        rho_s = 1.225                                   #air density sealevel
        rho_c = 0.26                                    #air density at cruise altitude
        P_cruise = 16235.7                              #air pressure cruise altitude
        T_cruise = 216.65                               #temperature at cruise altitude
        M_cruise = 0.85                                 #cruise mach number

        #output
        #cruise fuel fraction
        D_c = (pi*A*e)/(3*Cd0)
        LD_c = (3/4)*sqrt(D_c)
        num = R/((V/(g*Cj_c))*(LD_c))
        Wc = exp(num)
        Wcruise = 1/Wc
        W5 = round(Wcruise,5)

        # loiter fuel fraction
        D_l = 2*Cd0
        L_l = sqrt(pi*A*e*Cd0)
        LD_l = L_l/D_l
        T1 = 1/(g*Cj_l)
        end = T1*LD_l
        Wl =exp(E/end)
        W8 = round((1/Wl),5)

        # full fuel fraction method
        W1 = 0.990
        W2 = 0.995
        W3 = 0.995
        W4 = 0.980
        W6 = 0.990
        W7 = 0.992
        mtfo = 0.002
        a = 0.5153
        xy = 1063.2
        Wpl = 2563*9.80665

        # calculations
        mff = W1*W2*W3*W4*W5*W6*W7*W8
        Wmto = (xy+Wpl)/(mff-a-mtfo)
        We = a*Wmto + xy
        Wf = (1-mff)* Wmto
        Wfused = (1-(W1*W2*W3*W4*W5*W6*W7))*Wmto
        Wfres = Wf-Wfused
        Mres = Wfres/Wfused
        Wlanding = Wmto-Wfused
        f = Wlanding/Wmto
        Wcruise_begin = Wmto-((1-(W1*W2*W3*W4))*Wmto)
        Wcruise_end = Wmto-((1-(W1*W2*W3*W4*W5))*Wmto)
        
        run_iteration_002 = False

    #staring variables 
    mdd = 0.88

    C_l_cruise = (1.1/(0.5*rho_c*(V**2)))*(0.5*((Wcruise_begin+Wcruise_end)/s))
    
    s_l = 814.0                                     #landing distance [m]
    s_t = 1524.0                                    #take of distance [m]

    v_stall_landing = sqrt(s_l/0.5847)              #m/s
    v_stall_clean = 51.0                            #m/s
    v_cruise = 250.8                                #m/s

    K = 1/(A*e*pi)                                    
    C_D = Cd0 + (C_l_cruise**2)*K                   #drag polar
    TOP = 6641.24                                   #take of parameter


    f = (We+Wfres)/(Wmto)                           #fraction Woe/Wto
    climb_v = 0.22                                  #climb parameter set bij cs 25
    climb = 20.0                                    #climbgradient

    w_s = 0.1
    w_s_max = 0.001
    t_w = 1
    t_w_min = 1

    x_1 = 0.5*rho_s*C_l_landing*v_stall_landing**2
    x_2 = 0.5*rho_s*C_l_max_clean*v_stall_clean**2
    x_3 = (C_l_landing*rho_s*(s_l/0.5847))/(2*f)

    for i in range(0,6000000,1):
        w_s += 0.001
        if w_s <= x_1 and w_s <= x_2 and w_s <= x_3:
            if (w_s > w_s_max):
                w_s_max = w_s
     
    y_1 = (w_s_max/TOP)*1/C_l_take_of
    y_2 = ((rho_s/rho_c)**(3/4))*(((Cd0*0.5*rho_c*v_cruise**2)/(w_s_max))+w_s_max*((1)/(pi*A*e*0.5*rho_c*v_cruise**2)))
    y_3 = (climb)/(sqrt(w_s_max)*sqrt((2/rho_c)*(1/C_l_cruise))) + C_D/C_l_cruise
    y_4 = (C_D*0.5*rho_c*v_cruise**2)/(w_s_max)
    y_5 = climb_v + 2*sqrt((Cd0)/(pi*A*e))
                
    for j in range(0,100000,1):
        t_w -= 0.00001
        if t_w >= y_1 and t_w >= y_2 and t_w >= y_3 and t_w >= y_3 and t_w >= y_4 and t_w >= y_5:
            if (t_w < t_w_min):
                t_w_min = t_w
                
    #wing planform paramters 
    s = Wmto/w_s_max                                                #wing surface area [m^2]
    b = sqrt(s*A)                                                   #wing span [m]
    Lab1_4 = degrees(acos(0.75*(0.935)/(mdd)))                      #quater cord sweep angle [\deg]
    lab = 0.2*(2-Lab1_4*(pi/180))                                   #taper ratio
    c_r = (2*s)/(b*(1+lab))                                         #root cord [m]
    c_t = lab*c_r                                                   #tip cord [m]
    dihydral_angle = 3-(Lab1_4/10)+2                                #dyhidral angle [\deg]
    C_mac = (2/3)*c_r*((1+lab+lab**2)/(1+lab))                      #the mean aerodynamic cord [m]
    y_mgc = (b/6)*((1 +2*lab)/(1+lab))                              #y loaction for the mean aerodynamic cord
    tan_le = tan(radians(Lab1_4))-(c_r/(2*b))*(lab-1)               #tangent of leading edge sweepangle
    Le = degrees(atan(tan_le))                                      #leading edge sweep angle
    x_mgc = 10.3995 + 3.48 * ((1.392/3.48)*(0.091/0.405) -0.25*(1 + (0.091/0.405)))# tan_le*y_mgc                                            #x location for the mean aerodynamic cord
    Lab1_2 = degrees(atan(((b*tan_le)/(2)-0.5*c_r+0.5*c_t)/(b/2)))  #half cord sweep angle [deg]

    #airfoil
    # airfoil input parameters
    M_cross = 0.935
    radLab1_2 = radians(Lab1_2)
    mu = 1.3789*10**-5
    eff_fact = 0.95
    Beta = sqrt(1-M_cruise)
    Re_cruise = rho_c*V*C_mac/mu 

    #calculated parameters
    CLmax_Clmax = 0.75
    Cl_max = 2.243834
    aoa_zero_lift = 0
    delta_aoa_Cl_max = 4.2
    CD0 = 0.00601

    cgrad_rad = (2*pi*A)/(2+sqrt(4+((A*Beta/eff_fact)**2)*(1+(((tan(radLab1_2))**2)/(Beta**2)))))   # Lift gradient in rad( 1/rad)
    cgrad_deg = 1/(degrees(1/cgrad_rad))                                                            #Lift gradient in degrees( 1/deg)
    C_l_max_clean = CLmax_Clmax * Cl_max                                                            # maximum lift coefficient of the wing
    CL_max_grad = C_l_max_clean/cgrad_deg                  
    aoa_stall = CL_max_grad + aoa_zero_lift + delta_aoa_Cl_max                                      #stall angle of attack 
    CD_wing_cruise = CD0 + K*(C_l_cruise)**2                                                        # CD during cruise
 
    V_T = 0.54*((s**2)/b)*0.12*((1+lab+lab**2)/((1+lab)**2))
    v_f = (Wf/9.80665)/775

    if V_T <= v_f:
        print(V_T,v_f)
        print("problem")

    
    #ailerons
    #aileron requiremets
    banking = 45                                            #To be achieved banking angle
    t_reg = 1.4                                             #time requlation, sec
    run_aileron = True
    IN_board = 0.9899                                      #running variable
    tip_clearance = 0.1
    
    while run_aileron == True:

        d_var = 0.0001                   
        #Airfoil Characteristics, NACA...
        cl_alpha = cgrad_deg                                   #airfoil lift curve slope
                                             

        #Aileron characteristics & geometry
        b1 = IN_board * (b/2)                               #aileron starting point      
        b2 = (b/2) - tip_clearance                          #aileron ending point at a distance of tip_clearance from the end of the wing                        
        T = 0.56                                            #aileron effectiveness, obtained from graph
        deflection_up = 20                                  #degrees
        deflection_ratio = 0.75
        deflection_down = deflection_ratio * deflection_up  #degrees
        deflection = 0.5*(deflection_up + deflection_down)  #degrees
         
        chord_ratio = 0.35

        #Aileron Derivative, Cl_dalpha = k * integ
        constant = (2*cl_alpha*T)/(s*b)
        constant_linrel = (2*(lab-1))/b                                                                             #the constant from the c(y) relation
        integ1 = c_r*(((b2**2/2) + (constant_linrel*b2**3)/3) - (((b1**2/2) + (constant_linrel*b1**3)/3)))          #upper & lower bounds are b1,b2, c(y) integrated
        Cl_dalpha = constant * integ1


        #Roll damping coefficient
        constant2 = -((4*(cl_alpha + Cd0))/(s*b**2))
        integ2 = c_r *(b**3/24 + (constant_linrel*b**4)/64)
        Cl_p = constant2 * integ2

        #Roll rate
        P = -(Cl_dalpha/Cl_p) * deflection * ((2*V)/b)              
        dt = banking / P
            
        #chord, width, geometry
        width_aileron = b2 - b1                                   
        chord_wing_b1 = c_r*(1 + (2*(lab - 1)/b)*b1)
        chord_wing_b2 = c_r*(1 + (2*(lab - 1)/b)*b2)

        #area:
        chord_aileron_b1 = chord_ratio * chord_wing_b1
        chord_aileron_b2 = chord_ratio * chord_wing_b2
        S_ail = 0.5*width_aileron*(chord_aileron_b1 + chord_aileron_b2)
        
        if (t_reg*0.99)<= dt <= t_reg:
            run_aileron = False
        
        IN_board -= d_var
        
    #HLD design
    
    #calculation of reference area hld design
    D_outer = 2.9                                           #outer diameter of the fuselage
    d_clearance = 0.2                                       #clearance between fuselage or aileron with the hld
    d_tip_clearance = 0.1                                   #clearance between end leading edge flap and end wing
    A_constant = (c_r*(lab-1))/(b)                          #integration constant for wing area
    
    a_1 = D_outer/2 + d_clearance                           #distance from fuselage till start leading edge flap
    a_2 = (b/2) - d_tip_clearance                           #distance from the end of the leading edge flap till end of the wing
    Sref_leading = 2*(c_r*a_2 + A_constant*a_2**2 -(c_r*a_1 + A_constant*a_1**2))     #reference surface area leading edge flap

    b_1 = (D_outer/2)+d_clearance                           #distance from fuselage till start trailing edge flap
    b_2 = ((IN_board*b)/2)-d_clearance                      #distance between ending hld and start aileron
    Sref_trailing = 2*(c_r*b_2 + A_constant*b_2**2 -(c_r*b_1 + A_constant*b_1**2))    #reference surface area trailing edge flap

    chord_wing_b_1 = c_r*(1 + (2*(lab - 1)/b)*b_1)
    chord_wing_b_2 = c_r*(1 + (2*(lab - 1)/b)*b_2)
    
    # input data for the calculations
    Lab_065c = degrees(atan(((b*tan_le)/(2)-0.65*c_r+0.65*c_t)/(b/2)))      #hinge sweep angle for trailing edge HLD devices
    Lab_015c = degrees(atan(((b*tan_le)/(2)-0.15*c_r+0.15*c_t)/(b/2)))      #hinge sweep angle for leading edge HLD devices
    cos_hingline_trailing = cos(radians(Lab_065c))
    cos_hingline_leading = cos(radians(Lab_015c))
    
    C_c_takeof_f = 1.325                        #C_prime/C of a fowler flap at take of
    C_c_landing_f = 1.4095                      #C_prime/C of a fowler flap at landing

    C_c_takeof_df = 1.38675                     #C_prime/C of a double slotted fowler flap at take of
    C_c_landing_df = 1.572                      #C_prime/C of a double slotted fowler flap at landing

    aoa_zerolift_clean = aoa_zero_lift          # zero-lift angle of attack of the wing
    aoa_zerolift_land = - 15                    # zero-lift angle of attack dfference between clean and land config.
    aoa_zerolift_take = - 10                    # zero-lift angle of attack dfference between clean and take-off config.

    #average hld surface area
    s_flap = 0.5*(b_2-b_1)*((chord_wing_b_1*0.36725)+(chord_wing_b_2*0.36725))
    #s_flap = 0.5*(b_2-b_1)*((chord_wing_b_1*0.479375)+(chord_wing_b_2*0.479375))
    
    #take-of  data
    
    D_C_lmax_leading_t = 0.2                                        #delta clmax for a fixed slot
    D_C_lmax_trailing_t = 1.3 * C_c_takeof_f                        #delta clmax for a fowler flap
    S_prime_s_t = 1 + ((Sref_trailing)/(s))  * (C_c_takeof_f)
    D_flap_take_of = 15
   
    #D_C_lmax_leading_t = 0.3                                        #delta cl max for a leading edge flap
    #D_C_lmax_trailing_t = 1.6 * C_c_takeof_df                       #delta clmax for a double fowler flap
    #S_prime_s_t = 1 + ((Sref_trailing)/(s))  * (C_c_takeof_df)
    #D_flap_take_of = 20
    
    #landing data
    
    D_C_lmax_leading_l = 0.2                                        #delta clmax for a fixed slot
    D_C_lmax_trailing_l = 1.3 * C_c_landing_f                       #delta clmax for a fowler flap
    S_prime_s_l = 1 + ((Sref_trailing)/(s))  * (C_c_landing_f)
    D_flap_landing = 40
    used_flap = 'fowler flap'
    
    #D_C_lmax_leading_l = 0.3                                        #delta cl max for a leading edge flap
    #D_C_lmax_trailing_l = 1.6 * C_c_landing_df                      #delta clmax for a double fowler flap
    #S_prime_s_l = 1 + ((Sref_trailing)/(s))  * (C_c_landing_df)
    #D_flap_landing = 50
    #used_flap = 'double slotted fowler flap'
    
    #take of calculations
    D_C_LMAX_leading_t = 0.9 * D_C_lmax_leading_t * ((Sref_leading)/(s)) * cos_hingline_leading
    D_C_LMAX_trailing_t = 0.9 * D_C_lmax_trailing_t * ((Sref_trailing)/(s)) * cos_hingline_trailing
    D_C_LMAX_t = D_C_LMAX_leading_t + D_C_LMAX_trailing_t

    C_l_take_of = D_C_LMAX_t + C_l_max_clean

    D_a_0L_t = aoa_zerolift_take * ((Sref_trailing)/(s)) * cos_hingline_trailing
    a_0L_t = aoa_zerolift_clean + D_a_0L_t
    
    C_L_aflapped_t = S_prime_s_t * cgrad_deg
    
    #landing calculations
    D_C_LMAX_leading_l = 0.9 * D_C_lmax_leading_l * ((Sref_leading)/(s)) * cos_hingline_leading
    D_C_LMAX_trailing_l = 0.9 * D_C_lmax_trailing_l * ((Sref_trailing)/(s)) * cos_hingline_trailing
    D_C_LMAX_l = D_C_LMAX_leading_l + D_C_LMAX_trailing_l

    C_l_landing = D_C_LMAX_l + C_l_max_clean
    
    D_a_0L_l = aoa_zerolift_land * ((Sref_trailing)/(s)) * cos_hingline_trailing
    a_0L_l = aoa_zerolift_clean + D_a_0L_l
    
    C_L_aflapped_l = S_prime_s_l * cgrad_deg

    #empennage design
    #Weights
    WP = Wmto-We-Wf                     #N

    #Fuselage
    l_f = 18.36                         #m
    d_f_outer = 2.9                     #m
    d_f_inner = 2.82                    #m
    l_cabin = 9.72                      #m
    l_nosecone = 2 * d_f_outer          #m
    l_tailcone = 2.5 * d_f_outer        #m
    l_tail = 4.64                       #m
    l_cockpit = l_f - l_tail - l_cabin  #including bulkhead

    #Speed
    M_cruise = 0.85
    M_cross = 0.935
    sweepqc = acos(0.75*(M_cross/mdd))

    #Propulsion

    #CG excursion
    Mf_OEW = We/Wmto
    Mf_FUEL = Wf/Wmto
    Mf_PAYLOAD = WP/Wmto

    X_OEW = x_mgc + 0.25*C_mac          #m
    X_FUEL = x_mgc + 0.15*C_mac         #m
    X_PAYLOAD = l_cockpit + l_cabin/2

    X_OEWWP = (Mf_OEW*X_OEW + Mf_PAYLOAD*X_PAYLOAD)/(Mf_OEW + Mf_PAYLOAD)
    X_OEWWPWF = (Mf_OEW*X_OEW + Mf_PAYLOAD*X_PAYLOAD + Mf_FUEL*X_FUEL)/(Mf_OEW+Mf_PAYLOAD+Mf_FUEL)
    X_OEWWF = (Mf_OEW*X_OEW + Mf_FUEL*X_FUEL)/(Mf_OEW+Mf_FUEL)
    X_aftcg = max([X_OEWWP,X_OEW,X_OEWWF,X_OEWWPWF])

    #Empennage
    X_h = 15.7
    V_h = 0.67
    S_h = (V_h*s*C_mac)/(X_h-X_aftcg)
    sweepHqc = radians(Lab1_4)
    taperH = 0.65
    AH = 4.3
    bH = sqrt(S_h*AH)
    c_rH = (2*S_h)/((1+taperH)*bH)
    c_tH = taperH*c_rH
    MACH = c_rH*(2/3)*(1+taperH + taperH**2)/(1+taperH)

    X_v = 15
    V_v = 0.067
    S_v = (V_v*s*b)/(X_v-X_aftcg)
    sweepVLE = radians(43)
    taperV = 0.7
    AV = 1.1
    bV = sqrt(S_v*AV)
    c_rV = c_rH / taperV #(2*S_v)/((1+taperV)*bV)
    c_tV = c_rH #taperV*c_rV
    MACV = c_rV*(2/3)*(1+taperV + taperV**2)/(1+taperV)
    Lab1_4_v = degrees(atan(((bV*tan(sweepVLE))/(2)-0.25*c_rV+0.25*c_tV)/(bV/2)))


    
    
    stop_001 = abs(s_check-s)
    if stop_001<= 0.01 and iteration_check == False:
        run_iteration_001 = False
        

    if stop_001 <= 0.01 and iteration_check == True:
        iteration_check = False
    iteration_count += 1.0
    print("iteration:",iteration_count)
    print(s,Wmto,We)
    




print("\nTotal aircraft geometry desing")

#weight estimation
print("\nWeight estimation")
print("maximum take of weight =",round(Wmto,3))
print("operating empty weight =",round(We,3))
print("fuel weight =",round(Wf,3))

#design point
print("\nDesign Point")
print("Thrust over weight =",round(t_w_min,5))
print("Weight over surface area =",round(w_s_max,3))

#wing planform
print("\nWing planform")
print("Aspect ratio =",A)
print("wingsurface area =",round(s,2))
print("wingspan =",round(b,2))
print("root cord =",round(c_r,2))
print("tip cord =",round(c_t,2))
print("taper ratio =",round(lab,4))
print("leading edge sweep angle =",round(Le,1))
print("quarter cord sweep angle =",round(Lab1_4,1))
print("half cord sweep angle =", round(Lab1_2,1))
print("mean aerodynamic cord =",round(C_mac,2))
print("y position C_mac =",round(y_mgc,2))
print("x position (from leading edge) C_mac =",round(x_mgc,2))

#airfoil
print("\nAirfoil data")
print("The maximum lift coefficient is:",round(C_l_max_clean,3))
print("The lift gradient is:",round(cgrad_deg,4))
print("The zero-lift-angle of attack is:",aoa_zero_lift )
print("The stall angle of attack is :",round(aoa_stall,2))
print("The CD during cruise is :",round(CD_wing_cruise,5)) 


#aileron geometry
print("\nAileron geometry")
print("b1 =",round(b1,2))
print("b2 = ",round(b2,2))
print("b1 lies at,",round(IN_board,3),"of the wingspan")
print("The aileron control derivative is:", round(Cl_dalpha,5))
print("The roll damping is =", round(Cl_p,5))
print("The roll rate is =", round(P,3), "[deg/s]")
print("The time required is =", round(dt,5), "seconds")
print("width aileron =", round(width_aileron,2), "\nchord wing @b1 =", round(chord_wing_b1,2), "\nchord wing @b2 =", round(chord_wing_b2,2))
print("Chord aileron @b1 =", round(chord_aileron_b1,2), "\nchord aileron @b2 =", round(chord_aileron_b2,2))
print("Area of aileron is =", round(S_ail,2))

#HLD geometry   
print("\nHigh-lift device geometry",)
print("the used trailing edge flap type is,",used_flap)
print("the new max lift coefficient for landing is:",round(C_l_landing,3))
print("the flap deflection at landing is =",D_flap_landing)
print("the new max lift coefficient for take of is:",round(C_l_take_of,3))
print("the flap deflection at take of is =",D_flap_take_of)
print("the reference flapped surface are of TE is",round(Sref_trailing,2))
print("the reference flapped surface are of LE is",round(Sref_leading,2))
print("the new zero-lift angle of attack for take of is:",round(a_0L_t,2))
print("the new zero-lift angle of attack for landing is:",round(a_0L_l,2))
print("the new lift gradient for take of is =",round(C_L_aflapped_t,4))
print("the new lift gradient for landing is =",round(C_L_aflapped_l,4))

#emmapange geometry
print("\nemmanapge geometry")
print("\n vertical tail")
print("Aspect ratio vertical tail =",AV)
print("vertical tail surface area =",round(S_v,2))
print("vertical tail height =",round(bV,2))
print("root cord =",round(c_rV,2))
print("tip cord =",round(c_tV,2))
print("taper ratio =",round(taperV,4))
print("quarter cord sweep angle =",round(Lab1_4_v,1))
print("\nHorizontal tail")
print("Aspect ratio horizontal tail =",AH)
print("horizontal tail surface area =",round(S_h,2))
print("horizontal tail span =",round(bH,2))
print("root cord =",round(c_rH,2))
print("tip cord =",round(c_tH,2))
print("taper ratio =",round(taperV,4))
print("quarter cord sweep angle =",round(Lab1_4_v,1))

#fuselage geometry
print("\nFuselage geometry")
print("lenght of fuselage =",round(l_f,2))
print("inner diameter fuselage =",round(d_f_inner,2))
print("outer diameter fuselage =",round(d_f_outer,2))
print("lenght of the cabin =",round(l_cabin,2))
print("lenght of the nose cone =",round(l_nosecone,2))
print("lenght of the tail cone =",round(l_tailcone,2))
print("lenght of the tail =",round(l_tail,2))
print("lenght of the cockpit =",round(l_cockpit,2))
