    # Normalize the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude

        # Check the edges in the normal direction
        for n in range (int(width//2)):
            x2 = int(j + n * norm_x)
            y2 = int(i + n * norm_y)
            # Check the 8-connected neighbors around the current point  PEUT ETRE A MODIFIER EN FCT DE L ALGO FINAL
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = y2 + dx, x2+ dy
                    if nx >= 0 and nx < road.shape[0] and ny >= 0 and ny < road.shape[1] and width1<width//2:
                        road_copy[nx, ny] = [0, 255, 0]
            width1+=1

        for n in range (int(width//2)):
            x3 = int(j - n * norm_x)
            y3 = int(i - n * norm_y)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = y3 + dx, x3+ dy
                    if nx >= 0 and nx < road.shape[0] and ny >= 0 and ny < road.shape[1] and width2<width//2:
                        road_copy[nx, ny] = [0, 255, 0]
            width2+=1