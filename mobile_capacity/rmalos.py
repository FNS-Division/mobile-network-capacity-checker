import math
import logging

# Rural Macro-cell Model
class RMaLOS:

    def pathloss(hbs, hut, f, d, h):
        """
        Calculate Rural Macro-cell-Line-of-Sight Path Loss in dB.

        Parameters:
        - hbs (int): Height of BS antenna in meters.
        - hut (int): Height of UE transmitter in meters.
        - f (int): Frequency in GHz.
        - d (int): Distance in meters.  
        - h (int): Average height of buildings in meters.

        Returns:
        - pathloss (float): Rural Macro-cell-Line-of-Sight Path Loss in dB.

        Note:
        """
        # Breakpoint distance  
        dbp = (2 * math.pi * hbs * hut * f * (10 ** 9)) / (3 * (10 ** 8))
        # Rural Macro-cell Line of Sight Path loss before breakpoint distance in dB
        pl1 = 20*math.log(40*math.pi*d*f/3,10) + min(0.03*h**1.72,10) * math.log(d,10) - min(0.044*h**1.72,14.77) + 0.002*math.log(h,10)*d
        # Rural Macro-cell Line of Sight Path loss after breakpoint distance in dB
        pl2 = 20*math.log(40*math.pi*dbp*f/3,10) + min(0.03*h**1.72,10) * math.log(dbp,10) - min(0.044*h**1.72,14.77) + 0.002*math.log(h,10)*dbp + 40*math.log(d/dbp,10) 
        # if distance from BS to UE is less than BP distance then rmalos = pl1, elst rmalos = pl2
        if d <= dbp:
            pathloss = pl1
        else:
            pathloss = pl2
            
        return pathloss

    def ptxrb(bw, pnomref):
        """
        Calculate the transmit power per Resource Block (RB).

        Parameters:
        - bw (float): Bandwidth in MHz.
        - pnomref (float): Nominal reference power.
        - nrb (int): Number of resource blocks.

        Returns:
        - ptxrb (float): Transmit power per resource block in dB.

        Note:
        - One Resource Block spans 180 kHz in the frequency domain.
        - Due to guard bands and the need to fit an integer number of RBs 
        within the specified bandwidth, the actual number of RBs for 
        standard LTE bandwidths is fixed.
        """
        # logging.debug(f"Types - bw: {type(bw)}, pnomref: {type(pnomref)}, nrb: {type(nrb)}")

        # Check for bw and pnomref to be any numeric type
        if not all(isinstance(x, (float, int, np.number)) for x in [bw, pnomref]):
            logging.error(f"Non-numeric input: bw={bw}, pnomref={pnomref}")
            raise ValueError("bw and pnomref must be numeric.")

        # Constants
        RB_NUM_MULTIPLIER = 5

        # Adjust nrb based on bandwidth
        nrb = bw*RB_NUM_MULTIPLIER

        # Calculate Tx power per resource block
        ptxrb = 10*math.log(pnomref,10)-10*math.log(nrb,10) 

        # logging.info(f'bw = {bw}')
        # logging.info(f'pnomref = {pnomref}')
        # logging.info(f'nrb (initial) = {nrb}')
        # logging.info(f'nrb (adjusted) = {nrb}')
        # logging.info(f'ptxrb = {ptxrb}')

        return ptxrb

    def lpmax(ptxrb, gtx, grx, pathloss):
        """
        Calculate the received power at the Rx antenna.

        Parameters:
        - ptxrb (float): Transmit power per Resource Block in Watts.
        - gtx (int): Gain of the Tx antenna in dBi.
        - grx (int): Gain of the Rx antenna in dBi.
        - pathloss (float): RMa-LOS Path Loss in dB.

        Returns:
        - lpmax (float): Received power at the Rx antenna in dBm.

        Note:

        """
        # Check for bw and pnomref to be any numeric type
        if not all(isinstance(x, (float, int, np.number)) for x in [ptxrb, pathloss]):
            logging.error(f"Non-numeric input: ptxrb={ptxrb}, rmalos={pathloss}")
            raise ValueError("ptxrb and pathloss must be numeric.")

        # Check for nrb to be int or numpy integer type
        if not all(isinstance(x, (int, np.integer)) for x in [gtx, grx]):
            logging.error(f"Invalid input type for gtx or : gtx={gtx}, grx={grx}")
            raise ValueError("gtx and grx  must be an integer type.")

        # Calculate Received power at the Rx antenna in dBm.
        lpmax = ptxrb + gtx + grx - pathloss

        logging.info(f'lpmax = {lpmax}')

        return lpmax
