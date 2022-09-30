import numpy as np
import matplotlib.pyplot as plt

"""
Based on:
https://github.com/notadamking/Stock-Trading-Visualization
"""


class TSPGraph:
    """
    TSP rendering for the TSP environment
    """

    def __init__(self, window_size=10, time=0):
        """
        Initialise TSP matplotlib Graphs

        :param int window_size: window size of the distances
        """

        self.window_size = window_size
        # create a figure on screen and set the title
        fig = plt.figure()
        fig.suptitle('TSP')
        # create top subplot for net worth axis
        self.best_distance_ax = plt.subplot2grid((12, 1),
                                                 (0, 0),
                                                 rowspan=2,
                                                 colspan=1)

        # create bottom subplot for TSP plots
        self.tour_ax = plt.subplot2grid((12, 1),
                                        (3, 0),
                                        rowspan=4,
                                        colspan=1)

        # create 2nd bottom subplot for TSP plots
        self.best_tour_ax = plt.subplot2grid((12, 1),
                                             (8, 0),
                                             rowspan=4,
                                             colspan=1)

        # add padding to make graph easier to view
        plt.subplots_adjust(left=0.11,
                            bottom=0.24,
                            right=0.90,
                            top=0.90,
                            wspace=0.2,
                            hspace=1)



    def _render_distances(self, step_range):
        """
        Render distances

        :param list step_range: Steps to consider from the episode
        """

        # clear the frame rendered last step
        self.best_distance_ax.clear()
        self.best_distances = [x/10000 for x in self.best_distances]
        self.current_distances = [x/10000 for x in self.current_distances]
        # plot the distances
        self.best_distance_ax.plot(step_range,
                                   self.best_distances,
                                   '-', label='Best Distance',
                                   color = "darkred")
        self.best_distance_ax.plot(step_range,
                                   self.current_distances,
                                   '-', label='Current Distance', color="black")

        # show legend, which uses the labels defined above
        self.best_distance_ax.legend()
        legend = self.best_distance_ax.legend(loc=0, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        # save the last seen distance and the last step
        last_distance = self.current_distances[-1]
        best_distance = self.best_distances[-1]
        last_step = self.step_range[-1]

        # annotate the current distance on the graph
        self.best_distance_ax.annotate('{0:.2f}'.format(last_distance),
                                       (last_step, last_distance),
                                       xytext=(last_step, last_distance),
                                       bbox=dict(boxstyle='round',
                                       fc='w', ec='k', lw=1),
                                       color="black",
                                       fontsize=8)


       # annotate the current distance on the graph

        self.best_distance_ax.annotate('{0:.2f}'.format(best_distance),
                                       (last_step, best_distance),
                                       xytext=(last_step, best_distance),
                                       bbox=dict(boxstyle='round',
                                       fc='w', ec='k', lw=1),
                                       color="darkred",
                                       fontsize=8)


        # Add space above "max distance"
        self.best_distance_ax.set_ylim(0,
                                       max(np.maximum(self.best_distances,
                                                      self.current_distances) * 2))
        # plt.show(block=False)
        # self.fig.canvas.draw()
    def _render_tour(self, positions, axis):
        """
        Render TSP solutions in 2D

        :param np.array positions: Positions of (tour_len, 2) points
        """
        # clear the frame rendered last step
        # self.tour_ax.clear()
        axis.clear()
        # transform positions to np array
        # solution is eq. to traversing the graph in order
        solution = [x for x in range(positions.shape[0])]

        # plot scatters - red: depot
        # self.tour_ax.scatter(positions[:, 0], positions[:, 1])
        # self.tour_ax.scatter(positions[0, 0], positions[0, 1], color='darkred')


        axis.scatter(positions[:, 0], positions[:, 1])
        axis.scatter(positions[0, 0], positions[0, 1], color='darkred')

        #plot segments connecting the nodes and calculate tour distance
        start_node = 0
        distance = 0.
        N = len(solution)
        for i in range(N-1):

            start_pos = positions[start_node]
            next_node = solution[i + 1]
            end_pos = positions[next_node]

            # self.tour_ax.annotate("",
            #                       xy=start_pos, xycoords='data',
            #                       xytext=end_pos, textcoords='data',
            #                       arrowprops=dict(arrowstyle="-",
            #                                       connectionstyle="arc3"))

            axis.annotate("",
                                  xy=start_pos, xycoords='data',
                                  xytext=end_pos, textcoords='data',
                                  arrowprops=dict(arrowstyle="-",
                                                  connectionstyle="arc3"))

            distance += np.linalg.norm(end_pos - start_pos)
            start_node = next_node

        # self.tour_ax.annotate("",
        #                       xy=positions[start_node], xycoords='data',
        #                       xytext=positions[0], textcoords='data',
        #                       arrowprops=dict(arrowstyle="-",
        #                                       connectionstyle="arc3"))
        axis.annotate("",
                              xy=positions[start_node], xycoords='data',
                              xytext=positions[0], textcoords='data',
                              arrowprops=dict(arrowstyle="-",
                                              connectionstyle="arc3"))

        distance += np.linalg.norm(positions[start_node] - positions[0])
        textstr = "N nodes: %d\nTotal length: %.2f" % (N, distance)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # self.tour_ax.text(0.05, 0.95,
        #                   textstr,
        #                   transform=self.tour_ax.transAxes,
        #                   fontsize="small",
        #                   verticalalignment='top',
        #                   bbox=props)


        axis.text(0.05, 0.95,
                          textstr,
                          transform=axis.transAxes,
                          fontsize="small",
                          verticalalignment='top',
                          bbox=props)


    def render(self,
               current_step,
               best_distances,
               current_distances,
               state,
               best_state):
        """
        Render TSP distances and solutions in 2D

        :param int current_step: current episode step
        :param list best_distances: best distances found in an episode
        :param list current_distances: current distances in an episode
        :param torch.tensor state: positions of (tour_len, 2) points
        """

        self.best_distances = np.zeros(self.window_size)
        self.current_distances = np.zeros(self.window_size)

        window_start = max(current_step - self.window_size, 0)
        self.step_range = range(window_start, current_step + 1)
        s = slice(window_start, current_step + 1)
        self.best_distances = best_distances[s]
        self.current_distances = current_distances[s]

        self._render_distances(self.step_range)
        self._render_tour(state, self.tour_ax)
        self._render_tour(best_state, self.best_tour_ax)

        self.tour_ax.set_title('Current Tour', fontsize=8)
        self.best_tour_ax.set_title('Best Tour', fontsize=8)
        # hide duplicate labels

        self.tour_ax.get_xaxis().set_visible(False)
        plt.setp(self.best_distance_ax.get_xticklabels(), visible=False)
        # self.fig.canvas.draw()
        # Necessary to view frames before they are unrendered
        # plt.show(block=False)
        plt.pause(0.01)

        # show the graph without blocking the rest of the program


    def close(self):
        plt.close()
