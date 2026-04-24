import { createRouter, createWebHistory } from 'vue-router'
import DataUploadView from '../views/DataUploadView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'upload',
      component: DataUploadView
    },
    {
      path: '/pipeline',
      name: 'pipeline',
      component: () => import('../views/PipelineView.vue')
    },
    {
      path: '/results',
      name: 'results',
      component: () => import('../views/ResultsView.vue')
    }
  ]
})

export default router
