#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class GuidedFilter:
    def __init__(self, image, p, radius, eps):
        self.image = image
        self.p = p
        self.radius = radius
        self.eps = eps

    def custom_box(self, image, radius):
        (rows, cols) = image.shape[:2]
        result = np.zeros_like(image)

        tile = [1] * image.ndim
        tile[0] = radius
        cum_image = np.cumsum(image, 0)
        result[0:radius + 1, :,] = cum_image[radius:2 * radius + 1, :,]
        result[radius + 1:rows - radius, :,] = cum_image[2 * radius + 1:rows, :,] - cum_image[0:rows - 2 * radius - 1, :,]
        result[rows - radius:rows, :,] = np.tile(cum_image[rows - 1:rows, :,], tile) - cum_image[rows - 2 * radius - 1:rows - radius - 1, :,]

        tile = [1] * image.ndim
        tile[1] = radius
        cum_image = np.cumsum(result, 1)
        result[:, 0:radius + 1,] = cum_image[:, radius:2 * radius + 1,]
        result[:, radius + 1:cols - radius,] = cum_image[:, 2 * radius + 1:cols,] - cum_image[:, 0:cols - 2 * radius - 1,]
        result[:, cols - radius:cols, ...] = np.tile(cum_image[:, cols - 1:cols,], tile) - cum_image[:, cols - 2 * radius - 1:cols - radius - 1,]

        return result

    def custom_guided_filter_color(self):

        h, w = self.p.shape[:2]
        N = self.custom_box(np.ones((h, w)), self.radius)

        m_image_r = self.custom_box(self.image[:, :, 0], self.radius) / N
        m_iamge_g = self.custom_box(self.image[:, :, 1], self.radius) / N
        m_image_b = self.custom_box(self.image[:, :, 2], self.radius) / N

        m_p = self.custom_box(self.p, self.radius) / N

        m_image_p_r = self.custom_box(self.image[:, :, 0] * self.p, self.radius) / N
        m_image_p_g = self.custom_box(self.image[:, :, 1] * self.p, self.radius) / N
        m_image_p_b = self.custom_box(sefl.image[:, :, 2] * self.p, self.radius) / N

        cov_image_p_r = m_image_p_r - m_image_r * m_p
        cov_image_p_g = m_image_p_g - m_image_g * m_p
        cov_image_p_b = m_image_p_b - m_image_b * m_p

        var_image_rr = self.custom_box(self.image[:, :, 0] * self.image[:, :, 0], self.radius) / N - m_image_r * m_image_r
        var_image_rg = self.custom_box(self.image[:, :, 0] * self.image[:, :, 1], self.radius) / N - m_image_r * m_image_g
        var_image_rb = self.custom_box(self.image[:, :, 0] * self.image[:, :, 2], self.radius) / N - m_image_r * m_image_b

        var_image_gg = self.custom_box(self.image[:, :, 1] * self.image[:, :, 1], self.radius) / N - m_image_g * m_image_g
        var_image_gb = self.custom_box(self.image[:, :, 1] * self.image[:, :, 2], self.radius) / N - m_image_g * m_image_b

        var_image_bb = self.custom_box(self.image[:, :, 2] * self.image[:, :, 2], self.radius) / N - m_image_b * m_image_b

        a = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                sig = np.array([
                [var_image_rr[i, j], var_image_rg[i, j], var_image_rb[i, j]],
                [var_image_rg[i, j], var_image_gg[i, j], var_image_gb[i, j]],
                [var_image_rb[i, j], var_image_gb[i, j], var_image_bb[i, j]]])
                cov_image_p = np.array([cov_image_p_r[i, j], cov_image_p_g[i, j], cov_image_p_b[i, j]])
                a[i, j, :] = np.linalg.solve(sig + self.eps * np.eye(3), cov_image_p)

        b = m_p - a[:, :, 0] * m_image_r - a[:, :, 1] * m_image_g - a[:, :, 2] * m_image_b

        mean_A = self.custom_box(a, self.radius) / N
        mean_B = self.custom_box(b, self.radius) / N

        q = np.sum(mean_A * self.image, axis=2) + mean_B

        return q

    def custom_guided_filter_gray(self):
        
        (rows, cols) = self.image.shape
        N = self.custom_box(np.ones([rows, cols]), self.radius)

        mean_I = self.custom_box(self.image, self.radius) / N
        mean_P = self.custom_box(self.p.reshape(self.image.shape), self.radius) / N
        corr_I = self.custom_box(self.image * self.image, self.radius) / N
        corr_Ip = self.custom_box(self.image * self.p.reshape(self.image.shape), self.radius) / N
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_P

        a = cov_Ip / (var_I + self.eps)
        b = mean_P - a * mean_I

        mean_A = self.custom_box(a, self.radius) / N
        mean_B = self.custom_box(b, self.radius) / N

        q = mean_A * self.image + mean_B

        return q

    def custom_guided_filter_colorgray(self):
        if self.image.ndim == 2 or self.image.shape[2] == 1:
            return self.custom_guided_filter_gray()
        
        elif self.image.ndim == 3 and self.image.shape[2] == 3:
            return self.custom_guided_filter_color()
  
    def apply(self):
        if self.p.ndim == 2:
            self.p = self.p[:, :, np.newaxis]

        output = np.zeros_like(self.p)
        for ch in range(self.p.shape[2]):
            output[:, :, ch] = self.custom_guided_filter_colorgray()

        return np.squeeze(output) if self.p.ndim == 2 else output

